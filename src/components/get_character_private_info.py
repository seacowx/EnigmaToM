"""
Method to get characters' perception mask of entity state changes
"""

import os
import re
from numpy import character
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from copy import deepcopy
from components.utils import FileIO
from components.llms import OpenAIInference, OpenAIAsyncInference

# setting logging level for HuggingFace Transformers
from transformers.utils import logging
logging.set_verbosity_error()


class CharacterPrivateInfoGenerator:

    PREDEFINED_CATEGORIES = [
        'Peronsal Desire',
        'Personal Belief',
        'Personal Observation',
        'Personal Preference',
        'Personality Trait',
        'Personal Status',
    ]

    def __init__(
        self, 
        model: OpenAIInference,
        async_model: OpenAIAsyncInference,
        model_name: str,
        prompt_path: str, 
        dataset_name: str, 
    ) -> None:

        self.file_io = FileIO()
        self.message_template = self.__load_few_shot_examples(prompt_path, data_name=dataset_name)
        self.model = model
        self.model_name = model_name
        self.async_model = async_model


    def __load_few_shot_examples(
        self,
        prompt_path, 
        add_system_prompt: bool = False,
        data_name: str = 'bigtom',
    ) -> list:
        """
        load few-shot examples from the prompt bank
        grouped: keys are sentences and values are the corresponding character masks
        """

        prompt_bank = self.file_io.load_yaml(prompt_path)
        # data_prompt = prompt_bank[data_name]

        character_private_info_instruction = prompt_bank['character_private_info']

        message_template = []

        # if add_system_prompt:
        #     system_prompt = data_prompt['system_prompt']
        #     message_template.extend([{
        #             'role': 'system',
        #             'content': system_prompt
        #         }])


        # NOTE: load 3-shot examples

        # for i in range(1, 4):
        #     narrative = examples[i]
        #     cur_character_list_response = character_response[i]
        #     cur_character_private_info_response = private_info_responses[i]
        #
        #     for char in cur_character_list_response:
        #         character_private_info_prompt = character_private_info_instruction.replace('{{indexed narrative}}', narrative). \
        #             replace('{{character}}', char)
        #
        #         message_template.extend([
        #             {
        #                 'role': 'user',
        #                 'content': character_private_info_prompt
        #             },
        #             {
        #                 'role': 'assistant',
        #                 'content': str(cur_character_private_info_response[char])
        #             }
        #         ])

        message_template.append({
            'role': 'user',
            'content': character_private_info_instruction
        })

        return message_template


    @staticmethod
    def __get_digit(text: str) -> int:
        """
        Get contiguous digits from a string
        """

        try:
            detected_digit = int(''.join(re.findall(r'\d+', text)))
        except:
            detected_digit = -100

        return detected_digit


    def __parse_character_private_info(
        self, 
        result_lst: list, 
        character_lst: list, 
        indexed_narrative_lst: list
    ) -> list:

        """
        Parse LLM generated string to characters' perception masks in JSON
        Return list of dictionaries

        {characarter_name: private_info_indices}
        Example:
        {'Ava': [8], 'Logan': []} -> line 8 is privately accessible to Ava; Logan has no private info
        """

        idx = 0
        result_out = []
        for gt_chars, indexed_n in zip(character_lst, indexed_narrative_lst):

            for char in gt_chars:

                result = result_lst[idx]
                result = result.strip().split('\n')
                result = [ele.strip() for ele in result if ele.strip()]
                result = [
                    ele for ele in result if (
                        ele.startswith('-') or \
                        ele.startswith('*') or \
                        ele.startswith('•') or \
                        ele.startswith(u'\u2022') 
                    )
                ]

                result = [
                    ele.replace('-', ''). \
                        replace('*', ''). \
                        replace('•', ''). \
                        replace(u'\u2022', ''). \
                        strip() \
                    for ele in result 
                ]

                result_digit_temp, result_content = [], []
                for line in result:
                    try:
                        digit, content = line.split(':')
                        result_digit_temp.append(digit)
                        result_content.append(content)
                    except:
                        continue

                result_digit = []
                for digit in result_digit_temp:
                    try:
                        result_digit.append(int(digit))
                    except:
                        temp_digit = self.__get_digit(digit)
                        if temp_digit == -100:
                            result_digit.append(-100)
                        else:
                            result_digit.append(temp_digit)

                assert len(result_digit) == len(result_content)

                result_dict = {}
                for digit, content in zip(result_digit, result_content):
                    if digit == -100:
                        continue 

                    content = content.strip()

                    if digit not in result_dict.keys():
                        result_dict[digit] = {'private_info': content}
                    else:
                        result_dict[digit]['private_info'] = content

                result_out.append({char: result_dict})

                idx += 1

        return result_out


    async def generate(self, narrative: list, character_lst: list) -> tuple:

        character_private_info_msg_lst = []
        indexed_narrative_lst = []
        character_private_info_response_original = []
        for (indexed_n, chars) in zip(narrative, character_lst):

            complete_indexed_n = '\n'.join(indexed_n)

            for char in chars:
                character_private_info_msg = deepcopy(self.message_template)
                character_private_info_msg[-1]['content'] = character_private_info_msg[-1]['content'].replace('{{indexed narrative}}', complete_indexed_n) \
                    .replace('{{character}}', str(char))

                character_private_info_msg_lst.append(character_private_info_msg)

            indexed_narrative_lst.append(indexed_n)

        # character_private_info_response_original = []
        # for msg in tqdm(character_private_info_msg_lst):
        #     temp_out = self.model.inference(
        #         model=self.model_name,
        #         message=msg,
        #         temperature=0.,
        #     )
        #     character_private_info_response_original.append(temp_out)

        character_private_info_response_original = [
            self.async_model.inference(
                model=self.model_name,
                message=msg,
                temperature=0.,
            )
            for msg in character_private_info_msg_lst
        ]

        character_private_info_response_original = await atqdm.gather(*character_private_info_response_original)

        character_private_info_response = self.__parse_character_private_info(
            character_private_info_response_original, 
            character_lst,
            indexed_narrative_lst,
        )

        return character_private_info_response_original, character_private_info_response, character_private_info_msg_lst
