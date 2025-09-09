import os
import ast
import string
import random
import requests
import numpy as np
from copy import deepcopy
from tqdm.asyncio import tqdm

from transformers import AutoTokenizer
from components.utils import FileIO
from components.llms import OpenAIInference, OpenAIAsyncInference
from components.code_utils import exec_code, exec_private_code, extract_question_from_code

from make_character_state import init_state_builder
from construct_states.state_builder import StateBuilder


def convert_msg_to_jsonl(msg_lst: list, key_lst: list) -> list:

    counter = 0
    out_jsonl = []
    for cur_id, cur_msg in zip(key_lst, msg_lst):
        cur_json = {
            "custom_id": f"detect_location_{counter}-{cur_id}", 
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {
                "model": "gpt-4o", 
                "messages": cur_msg,
                "max_completion_tokens": 2000,
                "temperature": 0.,
            }
        }

        out_jsonl.append(cur_json)
        counter += 1

    return out_jsonl


class MaskToMEvaluator:


    # there are two eval methods
    # for datasets such as ToMi and OpenToM, use free-form eval
    # for datasets such as BigToM, which is designed to be evaluated with multiple-choice questions, use multiple-choice eval

    def __init__(
        self, 
        model_name: str, 
        data_name: str,
        to_jsonl: bool,
        no_async: bool = False, 
        init_model: bool = True
    ):
        self.file_io = FileIO()
        # model_name: (model_class, model_path, config_path, quantization)
        self.data_name = data_name
        self.model_name = model_name
        self.model_names = self.file_io.load_yaml('./configs/model_info.yml')
        self.to_jsonl = to_jsonl

        vllm_config = self.file_io.load_yaml('./configs/vllm_configs.yml')
        
        if not to_jsonl:

            # NOTE: check if the model server is up and running
            if 'gpt' not in model_name:
                cur_model_config = vllm_config[model_name]
            
                try:
                    attemp = requests.get(cur_model_config['base_url'])
                except:
                    raise Exception(
                        f'Initiate server at {cur_model_config["script_path"]} before running the script.'
                    )

                if attemp.status_code != 401:
                    raise Exception(
                        f'Initiate server at {cur_model_config["script_path"]} before running the script.'
                    )
                model_kwargs = dict(
                    base_url=cur_model_config['base_url'],
                    api_key=cur_model_config['api_key']
                )
            else:
                model_kwargs = self.file_io.load_yaml(os.path.expanduser('~/openai_info.yml'))

            self.no_async = no_async
            if init_model and not to_jsonl:
                if no_async:
                    self.model = OpenAIInference(
                        **model_kwargs
                    )
                else:
                    self.model = OpenAIAsyncInference(
                        **model_kwargs
                    )

    async def __make_simtom_narratives(
        self, 
        data: dict, 
        seed: int,
        model_name: str, 
        char_names: dict, 
        prompt: str,
    ) -> dict:

        simtom_msg_lst = []
        simtom_msg_key_lst = []
        for key, val in tqdm(data.items(), desc='Generating SimToM Narratives'):

            narrative = val['narrative'] + ' '
            narrative = narrative.split('. ')
            narrative = [ele for ele in narrative if ele.strip()]

            # narrative = augmented_data[key]['events']
            narrative = '\n'.join(narrative).strip()
            cur_char_names = char_names[key]

            for char in cur_char_names:
                cur_prompt = prompt.replace('{{narrative}}', narrative) \
                    .replace('{{character}}', char)

                cur_msg = [{
                    'role': 'user',
                    'content': cur_prompt,
                }]

                simtom_msg_lst.append(cur_msg)
                simtom_msg_key_lst.append(key)

        if self.to_jsonl:

            # NOTE: create directory for saving jsonl files
            cache_path = f'../data/openai_batch_exp_data/{self.model_name}'
            if not os.path.exists(
                cache_path
            ):
                os.makedirs(cache_path)

            cache_path = os.path.join(cache_path, self.data_name)
            if not os.path.exists(
                cache_path
            ):
                os.makedirs(cache_path)

            cache_path = os.path.join(cache_path, str(seed))
            if not os.path.exists(
                cache_path
            ):
                os.makedirs(cache_path)

            cache_path = os.path.join(
                cache_path,
                f'{self.data_name}_simtom_narrative_seed{seed}.jsonl'
            )

            result_cache_path = (
                f'../data/openai_batch_exp_processed/{self.model_name}/' 
                f'{self.data_name}/{seed}/{self.data_name}_simtom-narrative_seed{seed}.jsonl'
            )
            if os.path.exists(result_cache_path):
                result_lst = self.file_io.load_jsonl(result_cache_path)
                result_lst = [
                    ele['response']['body']['choices'][0]['message']['content'] for ele in result_lst
                ]
            else:
                out_jsonl = convert_msg_to_jsonl(
                    msg_lst=simtom_msg_lst,
                    key_lst=simtom_msg_key_lst,
                )
                self.file_io.save_jsonl(
                    obj=out_jsonl,
                    fpath=cache_path,
                )

        else:
            result_lst = []
            if self.no_async:
                for msg in tqdm(simtom_msg_lst):
                    temp_out = self.model.inference(
                        model=model_name,
                        message=msg,
                        temperature=0.,
                    )
                    result_lst.append(temp_out)
            else:
                result_lst = [self.model.inference(
                    model=model_name,
                    message=msg,
                    temperature=0.,
                ) for msg in simtom_msg_lst]
                result_lst = await tqdm.gather(*result_lst, desc='Generating SimToM Narratives')

        simtom_narratives = {}
        idx = 0
        for key, val in data.items():
            cur_char_names = char_names[key]
            simtom_narratives[key] = {char: '' for char in cur_char_names}

            for char in cur_char_names:

                simtom_narratives[key][char] = result_lst[idx].strip()

                idx += 1

        return simtom_narratives


    def __order_mc_questions(
        self, 
        question: dict, 
        candidate_keys: list, 
        candidate_delim: str,
        eval_methods: dict,
        representation: str,
        data_name: str = '',
    ) -> tuple:

        candidate_keys = eval_methods['eval_keys']
        candidate_delim = eval_methods['delimiter']

        if candidate_delim:
            candidate_lst = question[candidate_keys[0]].split(candidate_delim)
            candidate_lst = [ele.split('.', 1)[-1].strip() for ele in candidate_lst]
        else:
            # accessibility question from FANToM
            if 'wrong_answer' not in question:
                wrong_answer = 'yes' if question['correct_answer'] == 'no' else 'no'
                candidate_lst = [question['correct_answer'], wrong_answer]
            else:
                candidate_lst = [str(question[ele]) for ele in candidate_keys]

        candidate_lst = [ele.replace('_', ' ') for ele in candidate_lst]
        random.shuffle(candidate_lst)

        correct_ans = str(question[candidate_keys[1]]).replace('_', ' ')
        correct_idx = candidate_lst.index(correct_ans)

        if data_name == 'fantom' and (correct_ans in ['yes', 'no'] or '[' in correct_ans):
            if correct_ans in ['yes', 'no']:
                correct_letter = correct_ans
                candidate_str = '["yes", "no"]'
            elif '[' in correct_ans:
                correct_letter = ast.literal_eval(correct_ans)
                candidate_str = correct_letter + question['wrong_answer']
                candidate_str = str(list(set(candidate_str)))
        else:
            candidate_str = '\n'.join([
                f'{string.ascii_uppercase[i]}. {ele}' for i, ele in enumerate(candidate_lst)
            ])
            correct_letter = string.ascii_uppercase[correct_idx]

        return candidate_str, correct_letter

    
    def make_prompts(
        self, 
        prompt: str, 
        narrative: str, 
        question: dict,
        eval_method: str,
        eval_methods: dict,
        representation: str,
        data_name: str,
        masktom: bool = False,
        char_perspective: str = '',
        candidate_prompt: str = '',
    ) -> tuple:

        correct_letter = ''
        if eval_method == 'free-form':

            candidate_answers = ''
            if 'containers' in question:
                candidate_answers = question['containers']

            if masktom:
                if candidate_answers:
                    question_prompt = question['reduced_question'] + \
                        f' Choose one of the following options: {candidate_answers}'
                else:
                    question_prompt = question['reduced_question']

                cur_prompt = prompt.replace('{{narrative}}', narrative) \
                    .replace('{{question}}', question_prompt) \
                    .replace('{{character perspective}}', char_perspective) 

            else:
                if candidate_answers:
                    question_prompt = question['question'] + \
                        f' Choose one of the following options: {candidate_answers}'
                else:
                    question_prompt = question['question']

                cur_prompt = prompt.replace('{{narrative}}', narrative) \
                    .replace('{{question}}', question_prompt) 
        
        elif eval_method == 'multiple-choice':
            candidate_keys = eval_methods['eval_keys']
            candidate_delim = eval_methods['delimiter']

            candidate_str, correct_letter = self.__order_mc_questions(
                question=question,
                candidate_keys=candidate_keys,
                candidate_delim=candidate_delim,
                eval_methods=eval_methods,
                representation=representation,
                data_name=data_name,
            )

            if masktom:
                cur_prompt = prompt.replace('{{narrative}}', narrative) \
                    .replace('{{question}}', question['reduced_question']) \
                    .replace('{{multiple choice}}', candidate_str) \
                    .replace('{{character perspective}}', char_perspective) 

                # cur_prompt = prompt.replace('{{narrative}}', narrative) \
                #     .replace('{{question}}', question['question']) \
                #     .replace('{{multiple choice}}', candidate_str) \
                #     .replace('{{character perspective}}', char_perspective)

            else:
                cur_prompt = prompt.replace('{{narrative}}', narrative) \
                    .replace('{{question}}', question['question']) \
                    .replace('{{multiple choice}}', candidate_str) 

        elif eval_method == 'opentom':

            if masktom:
                cur_prompt = prompt.replace('{{narrative}}', narrative) \
                    .replace('{{question}}', question['reduced_question']) \
                    .replace('{{answer options}}', candidate_prompt) \
                    .replace('{{character perspective}}', char_perspective) 
            else:
                cur_prompt = prompt.replace('{{narrative}}', narrative) \
                    .replace('{{question}}', question['question']) \
                    .replace('{{answer options}}', candidate_prompt) 

        cur_msg = [{
            'role': 'user',
            'content': cur_prompt,
        }]

        return cur_msg, correct_letter
    

    def __make_simtom_prompts(
        self, 
        name_lst: list, 
        simtom_narratives: dict, 
        prompt, 
        backoff_prompt: str, 
        narrative: str,
        original_narrative: str,
        key: str, 
        question: dict,
        eval_method: str,
        eval_methods: dict,
        representation: str,
        data_name: str,
        candidate_prompt: str = '',
    ) -> tuple:

        if name_lst:
            character_narrative = simtom_narratives[key][name_lst[0]]

            prev_prompt = prompt[0].replace('{{narrative}}', narrative) \
                .replace('{{character}}', name_lst[0])

            correct_letter = ''
            if eval_method == 'free-form':

                if 'containers' in question:
                    containers = question['containers']
                    question_prompt = question['reduced_question'] + \
                        f' Choose one of the following options: {containers}'
                else:
                    question_prompt = question['reduced_question'] 

                cur_prompt = prompt[1].replace('{{character narrative}}', character_narrative) \
                    .replace('{{question}}', question_prompt) \
                    .replace('{{original narrative}}', original_narrative)
            
            elif eval_method == 'multiple-choice':
                candidate_keys = eval_methods['eval_keys']
                candidate_delim = eval_methods['delimiter']

                candidate_str, correct_letter = self.__order_mc_questions(
                    question=question,
                    candidate_keys=candidate_keys,
                    candidate_delim=candidate_delim,
                    eval_methods=eval_methods,
                    representation=representation,
                    data_name=data_name,
                )

                cur_prompt = prompt[1].replace('{{character narrative}}', character_narrative) \
                    .replace('{{question}}', question['question']) \
                    .replace('{{multiple choice}}', candidate_str) \
                    .replace('{{original narrative}}', original_narrative)

            elif eval_method == 'opentom':
                cur_prompt = prompt[1].replace('{{character narrative}}', character_narrative) \
                    .replace('{{question}}', question['reduced_question']) \
                    .replace('{{answer options}}', candidate_prompt) \
                    .replace('{{original narrative}}', original_narrative)

                # cur_prompt = prompt[1].replace('{{character narrative}}', character_narrative) \
                #     .replace('{{question}}', question['question']) \
                #     .replace('{{answer options}}', candidate_prompt)

            cur_msg = [
                {
                    'role': 'user',
                    'content': prev_prompt,
                },
                {
                    'role': 'assistant',
                    'content': narrative,
                },
                {
                    'role': 'user',
                    'content': cur_prompt,
                },
            ]

        else:
            # if no character name, backoff to normal prompt
            correct_letter = ''
            if eval_method == 'free-form':
                cur_prompt = backoff_prompt.replace('{{narrative}}', narrative) \
                    .replace('{{question}}', question['question'])
            
            elif eval_method == 'multiple-choice':
                candidate_keys = eval_methods['eval_keys']
                candidate_delim = eval_methods['delimiter']

                candidate_str, correct_letter = self.__order_mc_questions(
                    question,
                    candidate_keys,
                    candidate_delim,
                    eval_methods,
                    representation,
                )

                cur_prompt = backoff_prompt.replace('{{narrative}}', narrative) \
                    .replace('{{question}}', question['question']) \
                    .replace('{{multiple choice}}', candidate_str)

            elif eval_method == 'opentom':
                cur_prompt = backoff_prompt.replace('{{narrative}}', narrative) \
                    .replace('{{question}}', question['reduced_question']) \
                    .replace('{{answer options}}', candidate_prompt)
                # cur_prompt = backoff_prompt.replace('{{narrative}}', narrative) \
                #     .replace('{{question}}', question['question']) \
                #     .replace('{{answer options}}', candidate_prompt)

            cur_msg = [{
                'role': 'user',
                'content': cur_prompt,
            }]
        
        return cur_msg, correct_letter


    # def __retrieval_with_python(
    #     self,
    #     msg_lst: list, 
    #     result_lst: list,
    #     data_name: str, 
    #     model_name: str, 
    #     char_state_lst: list, 
    #     candidate_prompt_lst: list, 
    #     data_key_lst: list,
    # ) -> tuple:
    #
    #     """
    #     code for retrieving relevant info with llm-generated python code 
    #     """
    #
    #     combined_narrative_lst = []
    #     cur_question_lst = []
    #     candidate_prompt_parsed_lst = []
    #
    #     for idx, msg in enumerate(msg_lst):
    #         narrative_code = msg[-1]['content']
    #         narrative_code = narrative_code.split('```python', 1)[-1].split('```', 1)[0]
    #
    #         data_id = data_key_lst[idx]
    #
    #         if not os.path.isdir(f'../data/prelim_results/{data_name}/{model_name}/python_codes/{data_id}'):
    #             os.mkdir(f'../data/prelim_results/{data_name}/{model_name}/python_codes/{data_id}')
    #
    #         self.file_io.save_py(narrative_code.strip(), f'../data/prelim_results/{data_name}/{model_name}/python_codes/{data_id}/{idx}_narrative.py')
    #
    #         cur_question = extract_question_from_code(narrative_code)
    #
    #         temp_result = result_lst[idx]['generated_text']
    #
    #         retrieval_code = temp_result.split('```', 1)[-1].rsplit('```', 1)[0]
    #         retrieval_code = f'{narrative_code.strip()}\n\n{retrieval_code.strip()}'.replace('```', '').replace('```python', '')
    #
    #         self.file_io.save_py(retrieval_code, f'../data/prelim_results/{data_name}/{model_name}/python_codes/{data_id}/{idx}_retrieval.py')
    #
    #         private_narrative = char_state_lst[idx]['private_narrative']
    #         entity_base_class = char_state_lst[idx]['entity_base_class']
    #
    #         # generate facts publicly known to the character
    #         # generate information privately known to the character
    #         fact_narrative = exec_code(retrieval_code, char_state_lst[idx])
    #
    #         if private_narrative:
    #             private_narrative = exec_private_code(entity_base_class + '\n\n\n' + private_narrative)
    #
    #         combined_narrative = fact_narrative + '\n\n' + private_narrative
    #         candidate_prompt = ''
    #         if candidate_prompt_lst:
    #             candidate_prompt = candidate_prompt_lst[idx]
    #             candidate_prompt = candidate_prompt.split('=')[-1].strip()
    #
    #         combined_narrative_lst.append(combined_narrative)
    #         candidate_prompt_parsed_lst.append(candidate_prompt)
    #         cur_question_lst.append(cur_question)
    #
    #     return combined_narrative_lst, candidate_prompt_parsed_lst, cur_question_lst


    async def evaluate(
        self, 
        data: dict, 
        augmented_data: dict,
        data_name: str,
        char_names: dict, 
        prompt: str, 
        prompt_bank: dict,
        root_prompt_bank: dict,
        eval_methods: dict,
        masktom: bool,
        event_based: bool,
        seed: int,
        state_builder: StateBuilder | None = None,
        backoff_prompt: str = '',
        representation: str = '',
        eval_name: str = '',
        model_name: str = '',
    ) -> dict:

        simtom = False
        opentom_flag = False
        eval_method = eval_methods['eval_method']

        data_key_lst = list(data.keys())

        # NOTE: check if running SimToM prompting
        if isinstance(prompt, list):
            simtom = True

        if 'opentom' in data_name:
            opentom_flag = True
            candidate_answers = root_prompt_bank['candidate_answers']

            if representation == 'python':
                candidate_prompt = root_prompt_bank['python_answer_options']
            else:
                candidate_prompt = root_prompt_bank['answer_options']

        # if using SimToM, first collect all the character-centric narrative generations
        if simtom:
            simtom_narratives = await self.__make_simtom_narratives(
                data=data, 
                seed=seed,
                model_name=model_name,
                char_names=char_names, 
                prompt=prompt[0],
            )

        cur_msg_lst = []
        correct_letter_lst = []
        name_meta_lst = []
        char_state_lst = []
        candidate_prompt_lst = []
        msg_key_lst = []
        counter = 0
        
        skip_key_lst = []
        for key, val in data.items():

            if key in skip_key_lst:
                continue

            if event_based:
                narrative = augmented_data[str(key)]['events']
                narrative = '\n'.join(narrative)
            else:
                narrative = val['narrative']

            original_narrative = deepcopy(narrative)
            question_lst = val['questions']
            cur_char_names = char_names[key]

            if opentom_flag:
                plot_info = val['plot_info']
                containers = [plot_info['original_place'], plot_info['move_to_place']]

            for question in question_lst:

                cur_question = question['question']

                candidate_lst = []
                filled_candidate_prompt = ''
                if opentom_flag:
                    if 'initial location' in cur_question:
                        candidate_key = 'location_coarse'
                    elif 'precisely' in cur_question:
                        candidate_key = 'location_fine'
                    elif 'attitude' in cur_question:
                        candidate_key = 'attitude'
                    elif 'fullness' in cur_question:
                        candidate_key = 'fullness'
                    elif 'accessibility' in cur_question:
                        candidate_key = 'accessibility'
                    else:
                        candidate_key = 'location_fine'

                    candidate_lst = candidate_answers[candidate_key]

                    if not candidate_lst:
                        candidate_lst = containers

                    if candidate_lst:
                        candidate_lst = [f'"{ele}"' for ele in candidate_lst]
                        candidate_lst = ', '.join(candidate_lst)
                        filled_candidate_prompt = candidate_prompt.replace(
                            '{{answer options}}',
                            candidate_lst + ' '
                        ) + ' \n'

                cur_question = cur_question.split()
                cur_question = [
                    ele.lower().replace(',', '').replace('.', '').replace("'s", '')
                        for ele in cur_question
                ]

                name_idx_lst = []
                for name in cur_char_names:
                    if name.lower() in cur_question:
                        name_index = cur_question.index(name.lower())
                        name_idx_lst.append((name, name_index))

                name_idx_lst = sorted(name_idx_lst, key=lambda x: x[1])
                name_lst = [ele[0] for ele in name_idx_lst]
                name_meta_lst.append(name_lst)

                # build character-centric world state
                char_perspective = ''
                if state_builder:
                    if masktom:
                        try:
                            char_state, char_perspective = state_builder.build(key, name_lst)
                        except:
                            if key not in skip_key_lst:
                                skip_key_lst.append(key)
                            continue

                        char_state_lst.append(char_state)

                        narrative = char_state['narrative']
                        cur_question = question['reduced_question']

                    else:
                        char_state, char_perspective = state_builder.build(key, [])
                        narrative = char_state['narrative']
                        cur_question = question['question']

                else:
                    cur_question = question['question']

                # SimToM prompt, add llm-generated character-centric narrative
                # backoff to normal prompt when not a tom question (e.g. len(name_lst) == 0)
                if simtom:
                    cur_msg, correct_letter = self.__make_simtom_prompts(
                        name_lst=name_lst, 
                        simtom_narratives=simtom_narratives, 
                        prompt=prompt, 
                        backoff_prompt=backoff_prompt, 
                        narrative=narrative,
                        original_narrative=original_narrative,
                        key=key, 
                        question=question,
                        eval_method=eval_method,
                        eval_methods=eval_methods,
                        representation=representation,
                        candidate_prompt=filled_candidate_prompt,
                        data_name=data_name,
                    )

                else:
                    candidate_prompt_lst.append(filled_candidate_prompt)
                    cur_msg, correct_letter = self.make_prompts(
                        prompt=prompt,
                        char_perspective=char_perspective,
                        narrative=narrative,
                        question=question,
                        eval_method=eval_method,
                        eval_methods=eval_methods,
                        representation=representation,
                        masktom=masktom,
                        candidate_prompt=filled_candidate_prompt,
                        data_name=data_name,
                    )

                if correct_letter:
                    correct_letter_lst.append(correct_letter)

                cur_msg_lst.append(cur_msg)
                msg_key_lst.append(key)

                counter += 1

        if self.to_jsonl:

            # NOTE: create directory for saving jsonl files
            if not os.path.exists(
                f'../data/openai_batch_exp_data/{self.model_name}'
            ):
                os.makedirs(f'../data/openai_batch_exp_data/{self.model_name}')

            if not os.path.exists(
                f'../data/openai_batch_exp_data/{self.model_name}/{self.data_name}'
            ):
                os.makedirs(f'../data/openai_batch_exp_data/{self.model_name}/{self.data_name}')

            if not os.path.exists(
                f'../data/openai_batch_exp_data/{self.model_name}/{self.data_name}/{seed}/'
            ):
                os.makedirs(f'../data/openai_batch_exp_data/{self.model_name}/{self.data_name}/{seed}')

            # NOTE: convert msg_lst to jsonl format
            out_jsonl = convert_msg_to_jsonl(
                msg_lst=cur_msg_lst,
                key_lst=msg_key_lst,
            )
            
            self.file_io.save_jsonl(
                obj=out_jsonl,
                fpath=(
                    f'../data/openai_batch_exp_data/{self.model_name}/' 
                    f'{self.data_name}/{seed}/{eval_name}_seed{seed}.jsonl'
            ))

            self.file_io.save_json(
                obj=data,
                fpath=(
                    f'../data/openai_batch_exp_data/{self.model_name}/' 
                    f'{self.data_name}/{seed}/{eval_name}_data_data.json'
                )
            )

            if correct_letter_lst:
                self.file_io.save_json(
                    obj=correct_letter_lst,
                    fpath=(
                        f'../data/openai_batch_exp_data/{self.model_name}/' 
                        f'{self.data_name}/{seed}/{eval_name}_correct_letter_lst.json'
                    )
                )

            data = {}

        else:
            cur_result = []
            cur_narrative = []
            # extend data key lst, repeat each keye 9 times to account for 9 questions per data key

            if self.no_async:
                cur_result = []
                for msg in tqdm(cur_msg_lst):
                    temp_out = self.model.inference(
                        model=model_name,
                        message=msg,
                        temperature=0.,
                        max_tokens=512,
                    )
                    cur_result.append(temp_out)
            else:
                cur_result = [self.model.inference(
                    model=model_name,
                    message=msg,
                    temperature=0.,
                    max_tokens=512,
                ) for msg in cur_msg_lst]
                cur_result = await tqdm.gather(
                    *cur_result, 
                    desc=f'Evaluating | {data_name} | {model_name} | {eval_name} |'
                )

            if correct_letter_lst:
                assert len(correct_letter_lst) == len(cur_result), \
                    'Length mismatch between correct_letter_lst and cur_result'

            idx = 0
            for key, val in data.items():

                if key in skip_key_lst:
                    continue

                question_lst = val['questions']
                for j, question in enumerate(question_lst):

                    data[key]['questions'][j]['prompt'] = cur_msg_lst[idx][-1]['content']
                    data[key]['questions'][j]['predicted'] = cur_result[idx]
                    # data[key]['questions'][j]['prompt'] = cur_msg_lst[idx][-1]['content'].strip()

                    if cur_narrative:
                        data[key]['questions'][j]['short_narrative'] = cur_narrative[idx].strip()

                    if correct_letter_lst:
                        data[key]['questions'][j]['correct_letter'] = correct_letter_lst[idx]

                    idx += 1

                if simtom:
                    data[key]['simtom_narratives'] = simtom_narratives[key]

        return data


async def run_masktom(
    evaluator: MaskToMEvaluator,
    data_path: str,
    model: str,
    eval_name: str,
    world_state_model: str,
    prompt_path: str,
    seed: int,
    no_async: bool,
    event_based: bool,
    use_scene_graph: bool,
    prompt_type: str = 'vanilla-text',
    masktom: bool = False,
    masktom_version: str = 'ada',
    representation: str = 'text',
):
    '''
    run_exp run masktom evaluation experiments on tom datasets
    '''

    file_io = FileIO()
    data = file_io.load_json(data_path)

    if 'tomi' in data_path:
        data_name = 'tomi'
    elif 'bigtom' in data_path:
        data_name = 'bigtom'
    elif 'hitom' in data_path:
        data_name = 'hitom'
    elif 'opentom' in data_path:
        data_name = 'opentom'
    elif 'fantom_long' in data_path:
        data_name = 'fantom_long'
    elif 'fantom' in data_path:
        data_name = 'fantom'
    elif 'exploretom' in data_path:
        data_name = 'exploretom'
    else:
        raise ValueError('Current dataset is not supported')

    augmented_data = {}
    if event_based:
        augmented_data = file_io.load_json(
            f'../data/augmented_data/{model}/' 
            f'{data_name}_augmented_seed[{seed}].json'
        )

    # oracle_path = f"../data/tom_datasets/opentom/opentom_oracle_sample_{seed}.json"
    # if os.path.exists(oracle_path):
    #     oracle_data = file_io.load_json(oracle_path)

    data_keys = list(data.keys())

    supported_representations = ['text', 'markdown']
    special_representations = ['markdown']

    assert representation in supported_representations, (
        f'{representation} not supported. Please choose from {supported_representations}'
    )

    eval_methods = file_io.load_yaml('./configs/dataset_eval_config.yml')
    prompt_bank = file_io.load_yaml(prompt_path)
    eval_methods = eval_methods[data_name]
    eval_method = eval_methods['eval_method']
    root_prompt_bank = prompt_bank[eval_method]

    if masktom:
        prompt_bank = root_prompt_bank['masktom']
    else:
        prompt_bank = root_prompt_bank['original']

    if representation == 'markdown':
        prompt = prompt_bank['markdown']
    else:
        prompt = prompt_bank[prompt_type]
        
    backoff_prompt = ''
    if prompt_type == 'simtom-text':
        backoff_prompt = prompt_bank['vanilla-text']

    char_names = file_io.load_json(f'../data/masktom/ner/{data_name}_char_name_seed[{seed}].json')

    if 'fantom' in data_name or 'exploretom' in data_name:
        SAMPLE_SIZE = 50
    else:
        SAMPLE_SIZE = 100

    random.seed(seed)
    np.random.seed(seed)
    sampled_keys = np.random.choice(data_keys, SAMPLE_SIZE, replace=False)
    # sampled_keys = [k for k in sampled_keys if k in oracle_data.keys()]
    sampled_data = {key: val for (key, val) in data.items() if key in sampled_keys}
    
    state_builder = None
    if masktom or representation in special_representations:
        state_builder = init_state_builder(
            model=model,
            data_name=data_name,
            representation=representation,
            version=masktom_version,
            world_state_model=world_state_model,
            seed=seed,
            event_based=event_based,
            use_scene_graph=use_scene_graph,
        )

    result = await evaluator.evaluate(
        data=sampled_data, 
        augmented_data=augmented_data,
        data_name=data_name,
        char_names=char_names,
        prompt=prompt,
        prompt_bank=prompt_bank,
        root_prompt_bank=root_prompt_bank,
        eval_methods=eval_methods,
        event_based=event_based,
        masktom=masktom,
        state_builder=state_builder,
        backoff_prompt=backoff_prompt,
        representation=representation,
        eval_name=eval_name,
        model_name=model,
        seed=seed,
    )

    if result:
        postfix = ''
        postfix += f'_masktom_{masktom_version}' if masktom else ''
        postfix += '_event_based' if event_based else ''
        postfix += '_scene_graph' if use_scene_graph else ''

        root_dir = f'../data/prelim_results/{data_name}/{model}/{seed}'    
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        file_io.save_json(
            result, 
            os.path.join(
                root_dir, 
                f'{data_name}_{prompt_type}_{world_state_model}_{representation}{postfix}.json'
        ))
