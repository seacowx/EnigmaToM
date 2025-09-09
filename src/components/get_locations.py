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

import matplotlib

# setting logging level for HuggingFace Transformers
from transformers.utils import logging
logging.set_verbosity_error()


class LocationDetector():

    def __init__(
        self, 
        model: OpenAIInference | None,
        async_model: OpenAIAsyncInference | None,
        model_name: str,
        prompt_path: str, 
        dataset_name: str, 
        generation_config: dict,
    ) -> None:
        self.file_io = FileIO()

        if prompt_path and dataset_name:
            self.message_template, self.room_filter_prompt = self.__load_few_shot_examples(
                prompt_path=prompt_path, 
                data_name=dataset_name,
            )
        self.model = model
        self.async_model = async_model

        self.model_name = model_name
        self.generation_config = generation_config

        # define named colors for location filtering
        self.named_colors = list(matplotlib.colors.cnames.keys())


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

        narrative_location_instruction = 'identify_locations' # this prompt asks llm to generate all character masks at once
        room_filter_key = 'location_to_room'

        prompt_bank = self.file_io.load_yaml(prompt_path)
        general_prompt = prompt_bank

        narrative_location_instruction = general_prompt[narrative_location_instruction]

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
            'content': narrative_location_instruction
        })
        room_filter_message_template.append({
            'role': 'user',
            'content': room_filter_instruction
        })

        return message_template, room_filter_message_template


    def __get_room(self, room_lst: list) -> list:
        cur_msg = deepcopy(self.room_filter_prompt)
        cur_msg[-1]['content'] = cur_msg[-1]['content'].replace('{{location list}}', str(room_lst))

        result = self.model.inference(
            model=self.model_name,
            message=cur_msg, 
            temperature=0.,
        )

        try:
            result = result.split('\n')
            result = [ele.strip() for ele in result if ele.strip()]
            result = [ele for ele in result if ele.startswith('-')]
            result = [ele.replace('-', '').strip() for ele in result]
            result = [ele for ele in result if ele]
        except:
            result = []

        return result


    def parse_narrative_location(self, result_lst: list, data_name: str):
        """
        Parse LLM generated string to characters' perception masks in JSON
        """
        result_out = []

        for original_result in tqdm(result_lst, desc='Parsing narrative locations'):
            original_result = original_result.strip().lower()
            if data_name in ['bigtom']:
                original_result = original_result.rsplit('explicit locations:', 1)[-1]
            else:
                original_result = original_result.rsplit('rooms:', 1)[-1]

            original_result = original_result.split('\n')
            original_result = [ele for ele in original_result if '<|' not in ele]
            original_result = [ele.strip() for ele in original_result if ele.strip()]
            original_result = [ele.replace('-', '').replace('•', '').strip() for ele in original_result]
            original_result = [ele.strip().lower() for ele in original_result if ele]

            result = original_result
            result = [
                ele for ele in result if len(ele.split()) < 3
            ]

            if data_name in ['tomi', 'bigtom'] and len(result) > 1 and 'gpt' not in self.model_name:
                result = self.__get_room(result)

            # filter out location that contains words named colors
            result = [
                ele for ele in result if all([
                    tok not in self.named_colors for tok in ele.split()
            ])]

            result_out.append(result)

        return result_out


    async def detect(
        self, 
        narrative_lst: list, 
        data_name: str, 
        to_jsonl: bool,
        id_lst: list,
    ) -> tuple:

        # NOTE: generate character list
        narrative_location_msg_lst = []
        indexed_narrative_lst = []
        narrative_loc_result_original = []
        for indexed_n in narrative_lst:

            indexed_n = [ele.replace('_', ' ') for ele in indexed_n]
            complete_indexed_n = '\n'.join(indexed_n)

            # TODO: remove few-shot examples
            narrative_location_msg = deepcopy(self.message_template)
            cur_msg = narrative_location_msg[-1]['content'].replace(
                '{{indexed narrative}}', complete_indexed_n
            )

            if 'bigtom' in data_name:
                cur_msg = cur_msg.replace('room', 'explicit location') \
                    .replace('rooms', 'explicit locations') \
                    .replace('Rooms:', 'Explicit Locations:')

            narrative_location_msg[-1]['content'] = cur_msg

            narrative_location_msg_lst.append(narrative_location_msg)
            indexed_narrative_lst.append(complete_indexed_n)

        # save messages to JSONL for OpenAI batch processing
        jsonl_out = []
        if to_jsonl:
            counter = 0
            for i, cur_msg in enumerate(narrative_location_msg_lst):
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

            narrative_loc_result_parsed = []

        else:
            semaphore = asyncio.Semaphore(10)
            narrative_loc_result_original = [
                self.async_model.process_with_semaphore(
                    semaphore=semaphore,
                    model=self.model_name,
                    message=msg,
                    temperature=0.,
                ) for msg in narrative_location_msg_lst
            ]
            narrative_loc_result_original = await atqdm.gather(*narrative_loc_result_original)

            narrative_loc_result_parsed = self.parse_narrative_location(
                narrative_loc_result_original,
                data_name=data_name,
            )

        return narrative_loc_result_original, narrative_loc_result_parsed, narrative_location_msg_lst, jsonl_out
