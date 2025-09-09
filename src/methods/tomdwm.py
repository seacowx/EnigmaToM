import os
import re
import string
import random
from methods import masktom
import numpy as np

from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from components.utils import FileIO
from methods.masktom import MaskToMEvaluator
from components.llms import OpenAIInference, OpenAIAsyncInference
from make_character_state import init_state_builder


class ToMDwmEvaluator:

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
            to_jsonl=False,
            data_name=data_name,
        )
    
    @staticmethod
    def chunk_narrative(narrative, num_chunks=3):
        """
        Splits the narrative list into num_chunks evenly.

        Args:
            narrative (list): A list of strings representing events or sentences.
            num_chunks (int): The number of chunks to divide into (default is 3).

        Returns:
            list: A list containing num_chunks lists of strings.
        """
        # Calculate chunk sizes
        chunk_size = len(narrative) // num_chunks
        remainder = len(narrative) % num_chunks

        # Split into chunks, distributing the remainder evenly
        chunks = []
        start = 0
        for i in range(num_chunks):
            end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(narrative[start:end])
            start = end

        return chunks

    async def make_dwm_narratives(
        self, 
        data: dict, 
        model_name: str, 
        prompt: str, 
        instruction: str,
        history: dict,
        split_idx: int,
    ) -> dict:
        narrative_lst = []
        tomdwm_msg_lst = []
        for idx, row in enumerate(data):
            key = list(row.keys())[0]

            narrative = row[key]
            narrative = '\n'.join(narrative).strip()

            if history:
                narrative = f'{history[idx].strip()}\n\n{narrative}'

            cur_prompt = prompt.replace('{{dialogue chunk}}', narrative)
            cur_prompt += '\n' + instruction

            cur_msg = [{
                'role': 'user',
                'content': cur_prompt,
            }]

            tomdwm_msg_lst.append(cur_msg)
            narrative_lst.append(narrative)

        result_lst = [self.async_model.inference(
            model=model_name,
            message=msg,
            temperature=0.,
        ) for msg in tomdwm_msg_lst]
        result_lst = await atqdm.gather(*result_lst, desc=f'Generating ToMDwm Narratives | Split: {int(split_idx)+1}')

        history = []
        for idx, row in enumerate(narrative_lst):
            new_narrative = f"{row}\n{result_lst[idx].strip()}\n\n"
            history.append(new_narrative)

        return history


    async def evaluate(
        self,
        data: dict, 
        dwm_narrative_dict: dict,
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
        counter = 0
        for key, val in data.items():

            narrative = dwm_narrative_dict[key]

            question_lst = val['questions']
            cur_char_names = char_names[key]

            for question in question_lst:

                cur_question = question['question']

                candidate_lst = []
                filled_candidate_prompt = ''
                cur_question = cur_question.split()
                cur_question = [
                    ele.lower().replace(',', '').replace('.', '').replace("'s", '')
                        for ele in cur_question
                ]

                cur_question = question['question']

                candidate_prompt_lst.append(filled_candidate_prompt)
                cur_msg, correct_letter = self.masktom_evaluator.make_prompts(
                    prompt=prompt,
                    narrative=narrative,
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
        # extend data key lst, repeat each keye 9 times to account for 9 questions per data key
        extended_data_key_lst = [element for element in data_key_lst for _ in range(9)]

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

        return data


async def run_dwm(
    evaluator: ToMDwmEvaluator,
    data_path: str,
    model: str,
    prompt_path: str,
    dwm_prompt_path: str,
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
    elif 'opentom' in data_path:
        data_name = 'opentom'
    elif 'fantom_long' in data_path:
        data_name = 'fantom_long'
    elif 'fantom' in data_path:
        data_name = 'fantom'
    else:
        raise ValueError('Current dataset is not supported')

    if event_based:
        augmented_data = file_io.load_json(f'../data/augmented_data/{model}/{data_name}_augmented_seed[{seed}].json')

    data_keys = list(data.keys())

    eval_methods = file_io.load_yaml('./configs/dataset_eval_config.yml')
    prompt_bank = file_io.load_yaml(prompt_path)
    eval_methods = eval_methods[data_name]
    eval_method = eval_methods['eval_method']
    root_prompt_bank = prompt_bank[eval_method]

    prompt_bank = root_prompt_bank['original']
    prompt = prompt_bank['vanilla-text']

    char_names = file_io.load_json(f'../data/masktom/ner/{data_name}_char_name_seed[{seed}].json')

    tomdwm_prompt_bank = file_io.load_yaml(dwm_prompt_path)

    if 'fantom' in data_name:
        SAMPLE_SIZE = 50
    else:
        SAMPLE_SIZE = 100
    random.seed(seed)
    np.random.seed(seed)
    sampled_keys = np.random.choice(data_keys, SAMPLE_SIZE, replace=False)
    # sampled_keys = [k for k in sampled_keys if k in oracle_data.keys()]
    sampled_data = {key: val for (key, val) in data.items() if key in sampled_keys}

    chunks = {'0':[], '1':[], '2':[]}
    # original narrative
    for key, val in sampled_data.items():
        if event_based:
            narrative = augmented_data[id]['events']
        elif data_name == 'fantom':
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
        
        # now narrative is a list of strings List[str] where each string is an event 
        # you can now partition the narrative into 3 chunks and run DWM
        chunk = evaluator.chunk_narrative(narrative, num_chunks=3)
        chunks['0'].append({key: chunk[0]})
        chunks['1'].append({key: chunk[1]})
        chunks['2'].append({key: chunk[2]})

    quried_narratives = []

    for idx in chunks.keys():
        quried_narratives = await evaluator.make_dwm_narratives(
            data=chunks[idx], 
            model_name=model, 
            prompt=tomdwm_prompt_bank['construct_dwm'],
            instruction=tomdwm_prompt_bank['instruct_dwm'],
            history=quried_narratives,
            split_idx=idx,
        )

    # organize dwm narratives according to the original data ids
    dwm_narrative_dict = {}
    for idx, key in enumerate(sampled_data.keys()):
        dwm_narrative_dict[key] = quried_narratives[idx]

    result = await evaluator.evaluate(
        data=sampled_data,
        model_name=model,
        dwm_narrative_dict=dwm_narrative_dict,
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
            f'{data_name}_dwm{postfix}.json'
    ))
