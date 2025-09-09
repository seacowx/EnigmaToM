import ast
import asyncio
from tqdm import tqdm
from copy import deepcopy
from pydantic import BaseModel
from tqdm.asyncio import tqdm as atqdm
# from tqdm.asyncio import tqdm

from components.utils import FileIO
from components.llms import OpenAIInference, OpenAIAsyncInference


class EntityOfInterestExtractor:

    def __init__(
        self, 
        model: OpenAIInference | None,
        async_model: OpenAIAsyncInference | None,
        model_name: str,
        prompt_path: str, 
        dataset_name: str, 
        add_attr,
    ) -> None:
        self.file_io = FileIO()

        if prompt_path:
            self.message_template, self.revision_template = self.__load_few_shot_examples(
                prompt_path, 
                add_attr,
                data_name=dataset_name, 
            )

        self.model = model
        self.async_model = async_model
        self.model_name = model_name
        self.add_attr = add_attr


    def __load_few_shot_examples(
        self,
        prompt_path, 
        add_attr: bool,
        data_name: str,
        add_system_prompt: bool = False,
    ) -> tuple:
        """
        load few-shot examples from the prompt bank
        grouped: keys are sentences and values are the corresponding character masks
        """

        if add_attr:
            character_eoi_instruction = 'entity_of_interest_w_attribute'
            response_key = 'entity_of_interest_w_attribute'
        else:
            character_eoi_instruction = 'entity_of_interest' 
            response_key = 'entity_of_interest'

        # example_key = f'{data_name}_example'
        # question_key = f'example_questions'
        revision_key = 'revise_eoi_attr'
        # response_key = f'{data_name}_eoi_response_w_attribute'

        prompt_bank = self.file_io.load_yaml(prompt_path)

        character_eoi_instruction = prompt_bank[character_eoi_instruction]
        revise_instruction = prompt_bank[revision_key]

        # examples, questions, eoi_responses = data_prompt[example_key], data_prompt[question_key], data_prompt[response_key] 

        message_template = []

        if add_system_prompt:
            system_prompt = prompt_bank['eoi_system_prompt']
            message_template.extend([{
                    'role': 'system',
                    'content': system_prompt
                }])

        # for i in range(1, 4):
        #     narrative = examples[i]
        #     question = questions[i]
        #     question = '\n'.join([f'- {ele}' for ele in question])
        #     cur_eoi_response = eoi_responses[i]

        #     character_mask_prompt = character_eoi_instruction.replace('{{indexed narrative}}', narrative) \
        #         .replace('{{question list}}', question)

        #     message_template.extend([
        #             {
        #                 'role': 'user',
        #                 'content': character_mask_prompt
        #             },
        #             {
        #                 'role': 'assistant',
        #                 'content': str(cur_eoi_response)
        #             }
        #     ])

        message_template.append({
            'role': 'user',
            'content': character_eoi_instruction
        })

        return message_template, revise_instruction


    @staticmethod
    def __parse_single_eoi(cur_result: str, cur_char_lst: list) -> list:
        result = cur_result.split('<entities>')[1]
        result = result.split('\n')
        result = [ele.strip().lower() for ele in result if ele.strip()]

        dash_result = [ele for ele in result if ele.startswith('-')]
        asterisk_result = [ele for ele in result if ele.startswith('*')]
        dot_result = [ele for ele in result if ele.startswith('•')]

        result = dash_result + asterisk_result + dot_result

        result = [
            ele.replace('-', ''). \
                replace('*', ''). \
                replace('•', ''). \
                split('(', 1)[0].strip() for ele in result
        ]
        result = [' '.join(ele.split()[1:]) if ele.startswith('the ') else ele for ele in result]
        result = [ele.replace('_', ' ') for ele in result]
        result = list(set(result))

        result = [
            ele for ele in result if 
            all([ele.lower() not in char.lower() for char in cur_char_lst])
        ]

        return result


    def parse_eoi(
        self, 
        result_lst: list, 
        msg_lst: list, 
        char_lst: list, 
        add_attr: bool,
        do_correct: bool = True,
    ):
        """
        Parse LLM generated entities of interest
        """
        
        if do_correct and not msg_lst:
            raise ValueError((
                'Message list is empty. Cannot correct the generation.' 
                'Please provide the original message list.'
            ))

        result_out = []
        for idx, cur_result in enumerate(result_lst):

            if '<entities>' in cur_result:
                result = self.__parse_single_eoi(
                    cur_result=cur_result, 
                    cur_char_lst=char_lst[idx],
                )

            else:
                result = []

            # NOTE: Fix corrupted generation by rerunning llm with increased temperature
            if do_correct:
                temperature = 0.1
                TOLERANCE = 5
                counter = 0
                
                cur_msg = msg_lst[idx]

                while not result and counter < TOLERANCE:
                    cur_result = self.model.inference(
                        model=self.model_name,
                        message=cur_msg,
                        temperature=temperature,
                    )
                    if '<entities>' in cur_result:
                        result = self.__parse_single_eoi(
                            cur_result=cur_result,
                            cur_char_lst=char_lst[idx],
                        )
                    else:
                        result = []

                    temperature += 0.1
                    counter += 1

                if counter == TOLERANCE:
                    result = []

            result_out.append(result)

        return result_out


    async def __revise_eoi_attr(self, result_lst: list):

        msg_lst = []
        for result in result_lst:
            result = result.split('\n')
            result = [ele.strip() for ele in result if ele.strip()]
            dash_result = [ele for ele in result if ele.startswith('-')]
            asterisk_result = [ele for ele in result if ele.startswith('*')]

            result = dash_result + asterisk_result

            result = [ele.replace('-', '').replace('*', '').split('(', 1)[0].strip() for ele in result]
            result = list(set(result))

            revised_lst = []
            for ele in result:

                if ' of ' in ele:
                    revised_lst.append(ele)
                
                else:
                    cur_revision_prompt = self.revision_template.replace('{{eoi result}}', ele)

                    cur_revision_msg = [{
                        'role': 'user',
                        'content': cur_revision_prompt
                    }]

                    msg_lst.append(cur_revision_msg)

        # revised_result_lst = []
        # for msg in msg_lst:
        #     cur_response = self.model.inference(
        #         model=self.model_name,
        #         message=msg,
        #         temperature=0.,
        #     ) 
        #     revised_result_lst.append(cur_response)

        revised_result_lst = [self.async_model.inference(
            model=self.model_name,
            message=msg,
            temperature=0.,
        ) for msg in msg_lst]
        revised_result_lst = await atqdm.gather(*revised_result_lst)

        # revised_lst = '\n'.join([f'- {ele}' for ele in revised_lst])

        revised_result_lst = [ele.split(':')[-1].strip() for ele in revised_result_lst]

        return revised_result_lst


    async def extract(
        self, 
        narrative: list, 
        question_list: list, 
        char_lst: list,
        to_jsonl: bool,
        id_lst: list,
    ) -> tuple:

        # NOTE: generate character list
        eoi_msg_lst = []
        indexed_narrative_lst = []
        for (indexed_n, q) in zip(narrative, question_list):

            complete_indexed_n = '\n'.join(indexed_n)

            eoi_msg = deepcopy(self.message_template)
            eoi_msg[-1]['content'] = eoi_msg[-1]['content'].replace('{{indexed narrative}}', complete_indexed_n) \
                .replace('{{question list}}', q)

            eoi_msg_lst.append(eoi_msg)
            indexed_narrative_lst.append(complete_indexed_n)

        # eoi_result_original = []
        # for msg in tqdm(eoi_msg_lst):
        #     cur_response = self.model.inference(
        #         model=self.model_name,
        #         message=msg, 
        #         temperature=0.,
        #     )

        #     eoi_result_original.append(cur_response)
        jsonl_out = []
        if to_jsonl:
            counter = 0
            for i, cur_msg in enumerate(eoi_msg_lst):
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

            eoi_result_original, eoi_results = [], []
        else:
            semaphore = asyncio.Semaphore(10)

            eoi_result_original = [
                self.async_model.process_with_semaphore(
                    model=self.model_name,
                    semaphore=semaphore,
                    message=msg, 
                    temperature=0.,
            ) for msg in eoi_msg_lst]
            eoi_result_original = await atqdm.gather(*eoi_result_original)

            # if self.add_attr:
            #     revised_eoi_result = await self.__revise_eoi_attr(eoi_result_original)
            # else:
            #     revised_eoi_result = eoi_result_original

            eoi_results = self.parse_eoi(
                result_lst=eoi_result_original, 
                msg_lst=eoi_msg_lst,
                char_lst=char_lst,
                add_attr=self.add_attr,
            )

        return eoi_result_original, eoi_results, eoi_msg_lst, jsonl_out
