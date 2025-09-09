"""
Method to get characters' perception mask of entity state changes
"""

import os
import re
import time
import asyncio
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from typing import List
from copy import deepcopy
from components.utils import FileIO
from components.llms import OpenAIInference, OpenAIAsyncInference

import ast
import numpy as np

# setting logging level for HuggingFace Transformers
from transformers.utils import logging
logging.set_verbosity_error()


class CharacterLocationTracker():

    def __init__(
        self, 
        model: OpenAIInference | None,
        async_model: OpenAIAsyncInference | None,
        model_name:str,
        prompt_path: str, 
        dataset_name: str, 
        generation_config: dict, 
        seed: int,
    ) -> None:

        self.file_io = FileIO()

        if prompt_path:
            self.message_template, self.room_filter_prompt = self.__load_few_shot_examples(
                prompt_path, 
                data_name=dataset_name
            )

        self.model = model
        self.async_model = async_model

        self.location_reference_dict = self.file_io.load_json(
            f'../data/masktom/locations/{model_name}/{dataset_name}_samples_seed[{seed}].json'
        )
        self.model_name = model_name
        self.generation_config = generation_config

        self.ENTER_PATTERN = r'\b(enter|enters|entered)\b'
        self.EXIT_PATTERN = r'\b(exit|exits|exited)\b'


    def __load_few_shot_examples(
        self,
        prompt_path, 
        data_name: str,
        add_system_prompt: bool = False,
    ) -> tuple:
        """
        load few-shot examples from the prompt bank
        grouped: keys are sentences and values are the corresponding character masks
        """

        character_mask_instruction = 'character_location' # this prompt asks llm to generate all character masks at once
        room_filter_key = 'location_to_room'

        prompt_bank = self.file_io.load_yaml(prompt_path)
        general_prompt = prompt_bank

        character_mask_instruction = general_prompt[character_mask_instruction]
        room_filter_instruction = general_prompt[room_filter_key]

        message_template = []
        room_filter_message_template = []

        if add_system_prompt:
            system_prompt = general_prompt['system_prompt']
            message_template.extend([{
                    'role': 'system',
                    'content': system_prompt
                }])

        message_template.append({
            'role': 'user',
            'content': character_mask_instruction
        })
        room_filter_message_template.append({
            'role': 'user',
            'content': room_filter_instruction
        })

        return message_template, room_filter_message_template


    def parse_character_mask(self, response_lst: list) -> list:
        """
        Parse LLM generated string to characters' perception masks in JSON
        """

        out_lst = []
        for response in response_lst:
            original_result = deepcopy(response)

            result = response.split('\n')
            result = [ele.strip() for ele in result if ele.strip()]
            result = [ele for ele in result if '<|' not in ele]

            new_location_lst = []
            for line in result:
                cur_location_idx = line.split(':')[0].replace('-', '').replace('•', '').strip()
                if not cur_location_idx.isnumeric():
                    continue

                if re.search(self.ENTER_PATTERN, line, re.IGNORECASE):
                    cur_action = 'enter'
                    cur_location = re.split(self.ENTER_PATTERN, line, flags=re.IGNORECASE)[-1]
                    cur_location = cur_location.split('(')[0]
                    cur_location = cur_location.lower().strip()
                elif re.search(self.EXIT_PATTERN, line, re.IGNORECASE):
                    cur_action = 'exit'
                    cur_location = re.split(self.EXIT_PATTERN, line, flags=re.IGNORECASE)[-1]
                    cur_location = cur_location.split('(')[0]
                    cur_location = cur_location.lower().strip()
                else:
                    continue

                cur_location = ' ' + cur_location.strip('.') # Remove trailing period if present
                cur_location = cur_location.replace(' the ', '').strip()

                try:
                    new_location_lst.append({
                        'line_number': int(cur_location_idx),
                        'location': cur_location,
                        'action': cur_action,
                    })
                except:
                    continue

            if not new_location_lst:
                new_location_lst.append({
                    'line_number': -1,
                    'location': 'location',
                    'action': 'None',
                })

            out_lst.append(new_location_lst)

        return out_lst

    
    @staticmethod
    def make_location_vec(location_lst: list, narrative_len_lst: list):

        location_vec_lst = []

        for location, narrative_len in zip(location_lst, narrative_len_lst):

            if len(location) == 1 and location[0]['line_number'] == -1:
                location_vec_lst.append(['location'] * narrative_len)

            elif not location:
                location_vec_lst.append(['location'] * narrative_len)
            
            else:
                location_vec = [''] * narrative_len
                prev_line_idx = 0
                prev_location = 'none'
                prev_entered_location = ''
                prev_exited_line_idx = 0
                for location_state in location:

                    cur_line_idx, cur_location, cur_action = location_state.values()

                    if cur_action == 'enter':
                        cur_location = cur_location.replace('.', '').lower()
                        cur_location = cur_location.split('(', 1)[0].strip().lower()
                        prev_entered_location = cur_location
                    elif cur_action == 'exit':
                        cur_location = cur_location.replace('.', '').lower()
                        cur_location = cur_location.split('(', 1)[0].strip().lower()
                        # if the character exits the room without entering it,
                        # we need to fill the room with the previous location
                        if cur_location != prev_entered_location:
                            for idx in range(prev_exited_line_idx, cur_line_idx):
                                idx = min(idx, narrative_len - 1)
                                location_vec[idx] = prev_entered_location
                            continue
                        else:
                            cur_location = 'none'

                        prev_exited_line_idx = cur_line_idx

                    cur_line_idx = cur_line_idx - 1
                    if cur_line_idx >= narrative_len:
                        for idx in range(prev_line_idx, narrative_len):
                            location_vec[idx] = prev_location
                    else:
                        for idx in range(prev_line_idx, cur_line_idx):
                            location_vec[idx] = prev_location

                    prev_line_idx = cur_line_idx
                    prev_location = cur_location

                # handle the last entry
                for idx in range(prev_line_idx, narrative_len):
                    location_vec[idx] = prev_location

                if location_vec == ["none"] * narrative_len:
                    location_vec = ['location'] * narrative_len

                location_vec_lst.append(location_vec)

        return location_vec_lst
    

    async def track(self, narrative: list, character_list: list, id_lst: list, to_jsonl: bool) -> tuple:

        # NOTE: generate character list
        character_mask_msg_lst = []
        character_narrative_len_lst = []
        indexed_narrative_lst = []
        char_lst = []
        character_mask_results_original = []
        msg_id_lst = []
        for (id, complete_indexed_n, chars) in zip(id_lst, narrative, character_list):

            available_locations = self.location_reference_dict.get(id, [])
            if available_locations:
                available_locations = available_locations['detected_locations']

            available_locations = '\n'.join(['- ' + ele for ele in available_locations])

            for char in chars:
                character_mask_msg = deepcopy(self.message_template)
                character_mask_msg[-1]['content'] = character_mask_msg[-1]['content'] \
                    .replace('{{indexed narrative}}', complete_indexed_n) \
                    .replace('{{character}}', str(char)) \
                    .replace('{{location list}}', available_locations)

                character_narrative_len_lst.append(len(complete_indexed_n.split('\n')))

                character_mask_msg_lst.append(character_mask_msg)
                msg_id_lst.append(id)

            indexed_narrative_lst.append(complete_indexed_n)
            char_lst.extend(chars)

        # save messages to JSONL for OpenAI batch processing
        jsonl_out = []
        if to_jsonl:
            counter = 0
            for i, cur_msg in enumerate(character_mask_msg_lst):
                cur_json = {
                    "custom_id": f"detect_location_{counter}-{msg_id_lst[i]}", 
                    "method": "POST", 
                    "url": "/v1/chat/completions", 
                    "body": {
                        "model": "gpt-4o", 
                        "messages": cur_msg,
                        "max_completion_tokens": 2000,
                        "temperature": 0.,
                    }
                }
                jsonl_out.append(cur_json)
                counter += 1

            character_mask_results, character_mask_vecs = [], []

        else:
            semaphore = asyncio.Semaphore(10)
            character_mask_results_original = [
                self.async_model.process_with_semaphore(
                    semaphore=semaphore,
                    model=self.model_name,
                    message=msg,
                    temperature=0.,
                ) for msg in character_mask_msg_lst
            ]
            character_mask_results_original = await atqdm.gather(*character_mask_results_original)

            character_mask_results = self.parse_character_mask(
                response_lst=character_mask_results_original,
            )

            character_mask_vecs = self.make_location_vec(
                location_lst=character_mask_results,
                narrative_len_lst=character_narrative_len_lst,
            )

        return (
            character_mask_results_original, 
            character_mask_results, 
            character_mask_vecs, 
            character_mask_msg_lst,
            character_narrative_len_lst,
            jsonl_out,
        )
