"""
Method to get characters' perception mask of entity state changes
"""

import os
import re
import asyncio
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from copy import deepcopy
from components.utils import FileIO
from components.llms import OpenAIInference, OpenAIAsyncInference

# setting logging level for HuggingFace Transformers
from transformers.utils import logging
logging.set_verbosity_error()


class NarrativeLocationTracker():

    def __init__(
        self, 
        model: OpenAIInference | None,
        async_model: OpenAIAsyncInference | None,
        model_name: str,
        prompt_path: str, 
        dataset_name: str, 
        seed: int,
    ) -> None:
        self.file_io = FileIO()

        if prompt_path:
            self.message_template = self.__load_few_shot_examples(prompt_path, data_name=dataset_name)

        self.model = model
        self.async_model = async_model

        self.location_reference_dict = self.file_io.load_json(
            f'../data/masktom/locations/{model_name}/{dataset_name}_samples_seed[{seed}].json'
        )
        self.character_locations = self.file_io.load_json(
            f'../data/masktom/character_state/{model_name}/{dataset_name}_samples_seed[{seed}].json'
        )
        self.model_name = model_name


    def __load_few_shot_examples(
        self,
        prompt_path, 
        data_name: str,
        add_system_prompt: bool = False,
    ) -> list:
        """
        load few-shot examples from the prompt bank
        grouped: keys are sentences and values are the corresponding character masks
        """

        narrative_location_instruction = 'narrative_location' # this prompt asks llm to generate all character masks at once
        prompt_bank = self.file_io.load_yaml(prompt_path)
        general_prompt = prompt_bank

        narrative_location_instruction = general_prompt[narrative_location_instruction]

        message_template = []

        if add_system_prompt:
            system_prompt = general_prompt['system_prompt']
            message_template.extend([{
                    'role': 'system',
                    'content': system_prompt
                }])

        message_template.append({
            'role': 'user',
            'content': narrative_location_instruction
        })

        return message_template


    @staticmethod
    def parse_narrative_location(response: list) -> list:
        """
        Parse LLM generated string to characters' perception masks in JSON
        """

        response_lines = [ele for ele in response if ele.strip()]

        location_lst, location_idx_lst = [], []
        for idx, line in enumerate(response_lines):

            # skip corrupted generation
            try:
                cur_location_idx, cur_location = line.split(':')
            except ValueError:
                continue

            cur_location_idx = cur_location_idx.replace('-', '') \
                .replace('•', '').strip()

            cur_location = cur_location.strip()
            if not cur_location_idx.isnumeric():
                continue

            location_lst.append(cur_location)
            location_idx_lst.append(cur_location_idx)

        new_location_lst = []
        existing_locations = []
        prev_loc = ''
        for idx, loc in zip(location_idx_lst, location_lst):

            # if generation specify previous location and current location, only save current location
            if '->' in loc:
                loc = loc.split('->')[-1].strip()
            # if no location change, assume to be in the same room as previous event
            elif 'no room' in loc or 'no change' in loc or 'none' in loc:
                loc = prev_loc
            elif '(' in loc:
                loc = [ele for ele in existing_locations if ele in loc]
                if loc:
                    loc = loc[0]
                else:
                    loc = prev_loc

            # Fix the cases where some locations have underscore instead of space
            loc = loc.replace('_', ' ').strip().lower()

            if all(x.isalpha() or x.isspace() or x == "'" for x in loc):
                if loc not in existing_locations:
                    existing_locations.append(loc)
                prev_loc = loc

                new_location_lst.append({
                    'line_number': idx,
                    'location': loc
                })

                # correct previous locations
                for j, prev_loc_content in enumerate(new_location_lst):
                    if prev_loc_content['location'] == '{{FIX}}' or not prev_loc_content['location']:
                        new_location_lst[j]['location'] = loc
            else:
                new_location_lst.append({
                    'line_number': idx,
                    'location': '{{FIX}}'
                })

        return new_location_lst


    async def track(
        self, 
        id_lst: list, 
        narrative: list,
        data_name: str,
        to_jsonl: bool,
    ) -> tuple:

        # NOTE: generate character list
        narrative_location_msg_lst = []
        indexed_narrative_lst = []
        available_location_lst = []
        narrative_loc_result_original, narrative_loc_result_automatic = [], {}
        automatic_idx_lst = []
        for narrative_idx, (id, indexed_n) in enumerate(zip(id_lst, narrative)):

            complete_indexed_n = '\n'.join(indexed_n)

            llm_locations = self.location_reference_dict.get(id, [])
            
            cur_character_locations = list(
                set([ele for ele in self.character_locations if ele['id'] == id][0]['location_vec'])
            )

            if cur_character_locations == 'location':
                narrative_loc_result_original.append(
                    [f'- {idx+1}: location' for idx in range(len(indexed_n))]
                )
                indexed_narrative_lst.append(indexed_n)
                continue

            available_locations = []
            if (cur_llm_location := llm_locations['detected_locations']):
                available_locations = list(set(cur_llm_location + cur_character_locations))
                available_locations = [
                    ele.strip().lower() for ele in available_locations 
                    if ele.strip().lower() != 'none'
                ]
                available_locations = [ele for ele in available_locations if ele != 'location']

                # remove locations with special characters
                available_locations = [
                    ele for ele in available_locations if 
                        all(x.isalpha() or x.isspace() or x == "'" for x in ele)
                ]
                if len(available_locations) == 1 and 'bigtom' not in data_name:
                    narrative_location_msg_lst.append([{
                        'content': 'AUTOMATIC'
                    }])
                    narrative_loc_result_original.append(
                        [f'- {idx+1}: {available_locations[0]}' for idx in range(len(indexed_n))]
                    )

                    indexed_narrative_lst.append(indexed_n)

                    narrative_loc_result_automatic[narrative_idx] = [{
                        'line_number': idx,
                        'location': available_locations[0]
                    } for idx in range(len(indexed_n))]

                    available_location_lst.append(available_locations)
                    automatic_idx_lst.append(narrative_idx)
                    continue

            available_location_lst.append(available_locations)
            available_locations = '\n'.join(['- ' + ele for ele in available_locations])

            narrative_location_msg = deepcopy(self.message_template)
            narrative_location_msg[-1]['content'] = narrative_location_msg[-1]['content'].replace(
                '{{indexed narrative}}', complete_indexed_n
            ) \
                .replace('{{location list}}', available_locations)

            narrative_location_msg_lst.append(narrative_location_msg)
            indexed_narrative_lst.append(indexed_n)

        semaphore = asyncio.Semaphore(10)

        input_msg_lst = [
            ele for ele in narrative_location_msg_lst if 'AUTOMATIC' not in ele[0]['content']
        ]

        jsonl_out = []
        if to_jsonl:
            counter = 0
            for i, cur_msg in enumerate(input_msg_lst):
                cur_id = id_lst[i]
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
                jsonl_out.append(cur_json)
                counter += 1

            original_gen_out, parsed_gen_out = [], []
        else:
            result_lst = [
                self.async_model.process_with_semaphore(
                model=self.model_name,
                semaphore=semaphore,
                message=msg,
                temperature=0.,
            ) for msg in input_msg_lst]
            result_lst = await atqdm.gather(*result_lst)

            narrative_loc_result_original = [
                ele.split('\n') for ele in result_lst
            ]

            original_gen_out = []
            parsed_gen_out = []
            response_idx = 0
            for narrative_idx in range(len(id_lst)):
                if narrative_idx in automatic_idx_lst:
                    parsed_gen_out.append(
                        narrative_loc_result_automatic[narrative_idx]
                    )
                    original_gen_out.append('AUTOMATIC')
                else:
                    parsed_gen_out.append(
                        self.parse_narrative_location(
                            response=narrative_loc_result_original[response_idx],
                        ))
                    original_gen_out.append(narrative_loc_result_original[response_idx])
                    response_idx += 1

        return (
            original_gen_out, 
            parsed_gen_out, 
            narrative_location_msg_lst, 
            available_location_lst,
            jsonl_out,
            automatic_idx_lst,
            narrative_loc_result_automatic,
        )
