import os
import re
import ast
import string
import random
import asyncio
import numpy as np
from methods import masktom

from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from components.utils import FileIO
from methods.masktom import MaskToMEvaluator
from components.llms import OpenAIInference, OpenAIAsyncInference


class PerceptToMEvaluator:

    file_io = FileIO()

    def __init__(self, model_name: str, data_name: str) -> None:

        self.model_name = model_name

        if 'gpt' in model_name:
            model_kwargs = self.file_io.load_yaml(os.path.expanduser('~/openai_info.yml'))
        else:
            vllm_config = self.file_io.load_yaml('./configs/vllm_configs.yml')
            cur_model_config = vllm_config[model_name]

            model_kwargs = dict(
                base_url=cur_model_config['base_url'],
                api_key=cur_model_config['api_key']
            )

        self.model = OpenAIInference(
            **model_kwargs,
        )
        self.async_model = OpenAIAsyncInference(
            **model_kwargs,
        )

        self.masktom_evaluator = MaskToMEvaluator(
            model_name=model_name, 
            init_model=False,
            data_name=data_name,
            to_jsonl=False,
        )


    @staticmethod
    def find_first_character(question_words, character_names):
        char_names = [name.lower() for name in character_names]
            
        # Track earliest position of character mention
        first_char = ''
        first_pos = float('inf')
        
        for i, word in enumerate(question_words):
            for char in char_names:
                if char in word:
                    if i < first_pos:
                        first_pos = i
                        first_char = char
                        
        return first_char


    async def fix_corrupted_percettom_narrative(
        self,
        model_name: str,
        out_perception_dict: dict,
        temperature: float,
        fix_key_lst: list,
        fix_msg_lst: list,
    ):
        semaphore = asyncio.Semaphore(5)
        fix_perception_dict_lst = [
            self.async_model.process_with_semaphore(
                semaphore=semaphore,
                model=model_name,
                message=msg,
                temperature=temperature,
                max_tokens=4000,
            ) for msg in fix_msg_lst
        ]
        fix_perception_dict_lst = await atqdm.gather(*fix_perception_dict_lst)
        new_fix_key_lst, new_fix_msg_lst = [], []
        for idx, (response, key) in enumerate(zip(fix_perception_dict_lst, fix_key_lst)):
            try:
                response = ast.literal_eval(response)
                out_perception_dict[key] = response
            except:
                new_fix_key_lst.append(key)
                new_fix_msg_lst.append(fix_msg_lst[idx])

        return out_perception_dict, new_fix_key_lst, new_fix_msg_lst


    async def make_percepttom_narrative(
        self,
        data_name: str,
        perception_msg_dict: dict, 
        model_name: str, 
    ) -> dict:
        TOLERANCE = 5
        semaphore = asyncio.Semaphore(5)

        temperature = 0.
        # async inference does not work with FANToM + Llama3.3-70B
        if model_name == 'llama3-70b' and data_name == 'fantom':
            perception_dict_lst = []
            for msg in tqdm(perception_msg_dict.values()):
                response = self.model.inference(
                    model=model_name,
                    message=msg,
                    temperature=temperature,
                    max_tokens=6000,
                )
                perception_dict_lst.append(response)
        else:
            perception_dict_lst = [
                self.async_model.process_with_semaphore(
                    semaphore=semaphore,
                    model=model_name,
                    message=msg,
                    temperature=temperature,
                    max_tokens=5000,
                ) for msg in list(perception_msg_dict.values())
            ]
            perception_dict_lst = await atqdm.gather(*perception_dict_lst)

        perception_dict_key_lst = list(perception_msg_dict.keys())

        counter = 0
        out_perception_dict = {}
        fix_key_lst = []
        fix_msg_lst = []
        for idx, perception_dict in enumerate(perception_dict_lst):

            cur_key = perception_dict_key_lst[idx]

            if '```json' in perception_dict:
                perception_dict = perception_dict.split('```json')[1].split('```')[0].strip()
            elif '```' in perception_dict:
                temp_perception_dict = perception_dict.split('```')[1].strip()
                if temp_perception_dict:
                    perception_dict = temp_perception_dict.split('```')[0].strip()
                else:
                    perception_dict = perception_dict.split('```')[0].strip()

            try:
                perception_dict = ast.literal_eval(perception_dict)
                out_perception_dict[cur_key] = perception_dict
            except:
                fix_key_lst.append(cur_key)
                fix_msg_lst.append(perception_msg_dict[cur_key])
        
        if 'gpt' not in model_name:
            while fix_key_lst and counter < TOLERANCE:
                print(f'Fixing corrupted percepttom narrative | Attempt {counter+1}')
                temperature += 0.1
                out_perception_dict, fix_key_lst, fix_msg_lst = await self.fix_corrupted_percettom_narrative(
                    model_name=model_name,
                    out_perception_dict=out_perception_dict,
                    temperature=temperature,
                    fix_key_lst=fix_key_lst,
                    fix_msg_lst=fix_msg_lst,
                )
                counter += 1

            if counter == TOLERANCE and fix_key_lst:
                for key in fix_key_lst:
                    out_perception_dict[key] = []
        else:
            print(f"PerceptToM narratives are corrupted for {len(fix_key_lst)} samples")

        return out_perception_dict

    
    async def evaluate(
        self,
        data: dict, 
        percepttom_narratives: dict,
        original_narratives: dict,
        data_name: str,
        char_names: dict, 
        prompt: str, 
        root_prompt_bank: dict,
        eval_methods: dict,
        representation: str = '',
        eval_name: str = '',
        model_name: str = '',
    ):
        eval_method = eval_methods['eval_method']
        data_key_lst = list(data.keys())

        cur_msg_lst = []
        correct_letter_lst = []
        candidate_prompt_lst = []
        corrupted_key_lst = []
        counter = 0
        for key, val in data.items():

            try:
                percept_narrative = percepttom_narratives[key]
                original_narrative = original_narratives[key]
            except:
                corrupted_key_lst.append(key)
                continue

            question_lst = val['questions']
            cur_char_names = char_names[key]

            for question in question_lst:

                cur_question = question['question']

                candidate_lst = []
                filled_candidate_prompt = ''
                cur_question_tokens = cur_question.split()
                cur_question_tokens = [
                    ele.lower().replace(',', '').replace('.', '').replace("'s", '')
                        for ele in cur_question_tokens
                ]

                if percept_narrative:
                    # get the first character in the question and conduct perspective-taking
                    pivot_char = self.find_first_character(
                        question_words=cur_question_tokens,
                        character_names=cur_char_names,
                    )

                    if pivot_char:
                        pivot_char = pivot_char.strip()
                        input_narrative = []
                        for ele in percept_narrative:
                            cur_char_lst = list(ele.values())[0]
                            if pivot_char in cur_char_lst or pivot_char.capitalize() in cur_char_lst:
                                input_narrative.append(list(ele.keys())[0])
                    else:
                        input_narrative = [list(ele.keys())[0] for ele in percept_narrative]
                else:
                    # backoff to original narrative if percepttom narrative is corrupted
                    input_narrative = original_narrative

                input_narrative = ' '.join(input_narrative)

                candidate_prompt_lst.append(filled_candidate_prompt)
                cur_msg, correct_letter = self.masktom_evaluator.make_prompts(
                    prompt=prompt,
                    narrative=input_narrative,
                    question=question,
                    eval_method=eval_method,
                    eval_methods=eval_methods,
                    representation=representation,
                    candidate_prompt=filled_candidate_prompt,
                    data_name=data_name,
                )

                if correct_letter:
                    correct_letter_lst.append(correct_letter)

                cur_msg_lst.append(cur_msg)
                counter += 1

        cur_result = []
        cur_narrative = []
        cur_result = [self.async_model.inference(
            model=model_name,
            message=msg,
            temperature=0.,
        ) for msg in cur_msg_lst]
        cur_result = await atqdm.gather(
            *cur_result, 
            desc=f'Evaluating | {data_name} | {model_name} | {eval_name} |'
        )

        if correct_letter_lst:
            assert len(correct_letter_lst) == len(cur_result), \
                'Length mismatch between correct_letter_lst and cur_result'

        idx = 0
        for key, val in data.items():

            if key not in corrupted_key_lst:

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

        data = {key: val for key, val in data.items() if key not in corrupted_key_lst}

        return data


async def run_percepttom(
    evaluator: PerceptToMEvaluator,
    data_path: str,
    model: str,
    prompt_path: str,
    percepttom_prompt_path: str,
    event_based: bool,
    seed: int,
):
    '''
    run_exp run masktom evaluation experiments on tom datasets
    '''

    file_io = FileIO()
    data = file_io.load_json(data_path)

    # WARNING: make sure the dataset name is correct
    if 'tomi' in data_path:
        data_name = 'tomi'
    elif 'bigtom' in data_path:
        data_name = 'bigtom'
    elif 'hitom' in data_path:
        data_name = 'hitom'
    elif 'fantom' in data_path:
        data_name = 'fantom'
    else:
        raise ValueError('Current dataset is not supported')

    data_keys = list(data.keys())

    eval_methods = file_io.load_yaml('./configs/dataset_eval_config.yml')
    prompt_bank = file_io.load_yaml(prompt_path)
    eval_methods = eval_methods[data_name]
    eval_method = eval_methods['eval_method']
    root_prompt_bank = prompt_bank[eval_method]

    prompt_bank = root_prompt_bank['original']
    prompt = prompt_bank['vanilla-text']

    char_names = file_io.load_json(f'../data/masktom/ner/{data_name}_char_name_seed[{seed}].json')

    percepttom_prompt_bank = file_io.load_yaml(percepttom_prompt_path)

    if 'fantom' in data_name:
        SAMPLE_SIZE = 50
    else:
        SAMPLE_SIZE = 100
    random.seed(seed)
    np.random.seed(seed)
    sampled_keys = np.random.choice(data_keys, SAMPLE_SIZE, replace=False)
    # sampled_keys = [k for k in sampled_keys if k in oracle_data.keys()]
    sampled_data = {key: val for (key, val) in data.items() if key in sampled_keys}

    if 'fantom' in data_name:
        perception_prompt = percepttom_prompt_bank['fantom_get_perception']
    else:
        perception_prompt = percepttom_prompt_bank['get_perception']

    # original narrative
    perception_msg_dict = {}
    perception_narrative_lst_dict = {}
    for key, val in sampled_data.items():
        if data_name == 'fantom':
            narrative = val['narrative'].split('\n')
            narrative = [ele for ele in narrative if ele.strip()]
            narrative = [f'{idx+1}: {ele}.' for idx, ele in enumerate(narrative)]
        else:
            narrative = val['narrative'] + ' '
            narrative = narrative.replace('\n\n', ' ')
            narrative = narrative.replace('\n', ' ')
            narrative = narrative.split('. ')
            narrative = [ele for ele in narrative if ele.strip()]
            narrative = [f'{idx+1}: {ele}.' for idx, ele in enumerate(narrative)]

        perception_narrative_lst_dict[key] = narrative

        # concatenate indexed narrative into a string
        narrative = '\n'.join(narrative)
        cur_perception_prompt = perception_prompt.replace('{{narrative}}', narrative)

        perception_msg_dict[key] = [{'role': 'user', 'content': cur_perception_prompt}]

    perception_narrative_dict = await evaluator.make_percepttom_narrative(
        data_name=data_name,
        perception_msg_dict=perception_msg_dict,
        model_name=model, 
    )

    result = await evaluator.evaluate(
        data=sampled_data,
        model_name=model,
        percepttom_narratives=perception_narrative_dict,
        original_narratives=perception_narrative_lst_dict,
        data_name=data_name,
        char_names=char_names,
        prompt=prompt,
        root_prompt_bank=root_prompt_bank,
        eval_methods=eval_methods,
    )

    root_dir = f'../data/prelim_results/{data_name}/{model}/{seed}'    
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    postfix = '_event_based' if event_based else ''

    file_io.save_json(
        result, 
        os.path.join(
            root_dir, 
            f'{data_name}_percepttom{postfix}.json'
    ))
