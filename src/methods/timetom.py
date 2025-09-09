import os
import re
import string
import random
import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from components.utils import FileIO
from components.llms import OpenAIInference, OpenAIAsyncInference


class TimeToMEvaluator:

    file_io = FileIO()

    def __init__(
        self, 
        model_name: str, 
        use_blief_solver: bool, 
        to_jsonl: bool = False,
    ) -> None:

        self.use_belief_solver = use_blief_solver
        self.model_name = model_name
        self.to_jsonl = to_jsonl  

        if not self.to_jsonl:
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


    def __construct_temporal_space_msg(self, timetom_prompt_bank: dict, narrative: str) -> list:
        """
        Construct the temporal space of the narrative. 
        Section 2.1
        """

        temporal_state_prompt = timetom_prompt_bank['Constructing_Temporal_Space'].replace('{{story}}', narrative)

        temporal_state_prompt = [{
            'role': 'user',
            'content': temporal_state_prompt,
        }]

        return temporal_state_prompt


    def __construct_TBSC_msg(self, timetom_prompt_bank: dict, narrative_with_temporal_space: str, character: str) -> str:
        """
        Construct the TBSC of the narrative. 
        Section 2.2
        """

        tbsc_prompt = timetom_prompt_bank['Temporal_Belief_State_Chain_Construction'].replace('{{character}}', character) \
            .replace('{{story}}', narrative_with_temporal_space)

        tbsc_prompt = [{
            'role': 'user',
            'content': tbsc_prompt,
        }]

        return tbsc_prompt


    def __belief_compression(self, timetom_prompt_bank: dict, character_tbsc: str, character: str) -> list:
        """
        Belief compression of the narrative. 
        Section 2.3
        """

        belief_compression_prompt = timetom_prompt_bank['Belief_Compression'].replace('{{perspective}}', character_tbsc) \
            .replace('{{character}}', character)

        belief_compression_prompt = [{
            'role': 'user',
            'content': belief_compression_prompt,
        }]

        return belief_compression_prompt


    def __make_question_prompts(
        self, 
        question: dict, 
        candidate_keys: list, 
        candidate_delim: str,
        prompt_bank: dict,
        containers: list,
        mc_probing: bool = True,
    ) -> tuple:

        if candidate_delim:
            candidate_lst = question[candidate_keys[0]].split(candidate_delim)
            candidate_lst = [ele.split('.', 1)[-1].strip() for ele in candidate_lst]
        else:
            try:
                candidate_lst = [question[ele] for ele in candidate_keys]
            except:
                # Binary question in FANToM does not have candidate keys
                candidate_lst = ['yes', 'no']

        random.shuffle(candidate_lst)
        correct_ans = question[candidate_keys[1]]
        correct_idx = candidate_lst.index(correct_ans)

        if isinstance(candidate_lst[0], list):
            candidate_lst = sum(candidate_lst, [])

        if mc_probing:
            candidate_str = '\n'.join([
                f'{string.ascii_uppercase[i]}. {ele}' for i, ele in enumerate(candidate_lst)
            ])
            out_question = question['question'] + '\n' + candidate_str
            out_reduced_question = question['reduced_question'] + '\n' + candidate_str
            correct_letter = string.ascii_uppercase[correct_idx]
        else:
            candidate_str = str(candidate_lst)
            out_question = question['question'] + ' Choose one of the following options: ' + candidate_str
            out_reduced_question = question['reduced_question'] + ' Choose on of the following options: ' + candidate_str
            correct_letter = correct_ans

        return out_question, out_reduced_question, correct_letter


    def __first_order_qa_msg(
        self, 
        timetom_prompt_bank: dict, 
        compressed_narrative: str, 
        character: str, 
        question_prompt: str,
        character_tbsc_dict: dict,
        question_type: str = '',
    ) -> list:

        all_character_beliefs = ''
        for key_char, tbsc in character_tbsc_dict.items():
            all_character_beliefs += f'{key_char}:\n{tbsc}\n\n'

        all_character_beliefs = all_character_beliefs.strip()

        character_tbsc = ''
        if character != 'omniscent-view':
            character_tbsc = character_tbsc_dict[character]

        # QA prompt for Dialogue (FANToM)
        fact_info = ''
        if 'Information:' in question_prompt:
            fact_info, question_prompt = question_prompt.split('\n')
            fact_info = fact_info.replace('Information:', '').strip()
        elif 'Question:' in question_prompt:
            fact_info, question_prompt = question_prompt.split('\n')
            fact_info = fact_info.replace('Question:', '').strip()

        if question_type == 'answerability_binary':
            qa_prompt = timetom_prompt_bank['Time-Aware_Answerability_Binary_Question_Answer']
        elif question_type == 'infoaccess_binary':
            qa_prompt = timetom_prompt_bank['Time-Aware_Infoaccess_Binary_Question_Answer']
        elif question_type == 'answerability_list':
            qa_prompt = timetom_prompt_bank['Time-Aware_Answerability_List_Question_Answer']
        elif question_type == 'infoaccess_list':
            qa_prompt = timetom_prompt_bank['Time-Aware_Infoaccess_List_Question_Answer']
        else:
            qa_prompt = timetom_prompt_bank['Time-Aware_Belief_QA']

        qa_prompt = qa_prompt.replace('{{compressed perspective}}', compressed_narrative) \
            .replace('{{perspective}}', compressed_narrative) \
            .replace('{{character}}', character) \
            .replace('{{question}}', question_prompt) \
            .replace('{{all character beliefs}}', all_character_beliefs) \
            .replace('{{fact info}}', fact_info) \
            .replace('{{character belief}}', character_tbsc)

        qa_msg = [{
            'role': 'user',
            'content': qa_prompt,
        }]

        return qa_msg


    @staticmethod
    def __extract_time_stamps(narrative_with_temporal_space: str, return_narrative: bool = False) -> list:
        nar_lst = narrative_with_temporal_space.split('\n')
        nar_lst_first_token = [ele.split(' ', 1)[0] for ele in nar_lst]
        nar_lst_last_token = [ele.rsplit(' ', 1)[-1] for ele in nar_lst]

        nar_lst_token = []
        if any(re.search(r't\d', token) for token in nar_lst_first_token):
            nar_lst_token = nar_lst_first_token
        if any(re.search(r't\d', token) for token in nar_lst_last_token):
            nar_lst_token = nar_lst_last_token

        clean_nar_lst_token = []
        valid_idx = []
        for idx, token in enumerate(nar_lst_token):
            if re.search(r't\d', token):
                clean_nar_lst_token.append(token.replace(':', '').replace(',', ''))
                valid_idx.append(idx)

        if return_narrative:
            valid_narrative = [ele for idx, ele in enumerate(nar_lst) if idx in valid_idx]
            output_lst = []
            for token, nar in zip(clean_nar_lst_token, valid_narrative):
                output_lst.append([token, nar])
            return output_lst
            
        else:
            return clean_nar_lst_token
    

    def __belief_solver_stage1_msg(
        self, 
        timetom_prompt_bank: dict, 
        narrative_with_temporal_space: str, 
        sorted_char_lst: list,
        character_tbsc_dict: dict,
        question: str,
        reduced_question: str,
    ) -> list:

        solver_prompt_lst = timetom_prompt_bank['Time-Aware_Belief_Solver']
        character_prompt = ''
        if len(sorted_char_lst) == 2:
            character_prompt = f'{sorted_char_lst[0]} and {sorted_char_lst[1]}'
        else:
            for idx, char in enumerate(sorted_char_lst):

                if idx == len(sorted_char_lst) - 1:
                    character_prompt += f'and {char}'
                else:
                    character_prompt += f'{char}, '

        character_tbsc = character_tbsc_dict[sorted_char_lst[-1]]

        # stage 1 prompt
        cur_prompt = solver_prompt_lst[0]
        cur_prompt = cur_prompt.replace('{{perspective}}', narrative_with_temporal_space) \
            .replace('{{character}}', sorted_char_lst[-1]) \
            .replace('{{question}}', reduced_question) \
            .replace('{{character belief}}', character_tbsc)
        
        cur_msg = [{
            'role': 'user',
            'content': cur_prompt,
        }]

        return cur_msg


    def __belief_solver_stage2_msg(
        self, 
        timetom_prompt_bank: dict, 
        common_narrative: str,
        sorted_char_lst: list,
        question: str,
        reduced_question: str,
    ) -> list:

        # stage 2 prompt
        solver_prompt_lst = timetom_prompt_bank['Time-Aware_Belief_Solver']
        character_prompt = ''
        if len(sorted_char_lst) == 2:
            character_prompt = f'{sorted_char_lst[0]} and {sorted_char_lst[1]}'
        else:
            for idx, char in enumerate(sorted_char_lst):

                if idx == len(sorted_char_lst) - 1:
                    character_prompt += f'and {char}'
                else:
                    character_prompt += f'{char}, '

        cur_prompt = solver_prompt_lst[1]
        cur_prompt = cur_prompt.replace('{{character list}}', character_prompt) \
            .replace('{{common belief}}', common_narrative) \
            .replace('{{question}}', reduced_question) \
            .replace('{{character1}}', sorted_char_lst[0]) \
            .replace('{{character2}}', sorted_char_lst[1]) 

        if len(sorted_char_lst) > 2:
            cur_prompt = cur_prompt.replace('{{character3}}', sorted_char_lst[2])
        else:
            cur_prompt = cur_prompt.replace('{{character3}}', 'others')

        cur_msg = [{
            'role': 'user',
            'content': cur_prompt,
        }]

        return cur_msg


    def __belief_solver_stage3_msg(
        self, 
        timetom_prompt_bank: dict, 
        narrative_with_temporal_space: str, 
        common_narrative: str,
        sorted_char_lst: list,
        question: str,
        reduced_question: str,
        character_tbsc_dict: dict,
        temp_answer1: str = '{{answer1}}',
        temp_answer2: str = '{{answer2}}',
    ) -> list:

        solver_prompt_lst = timetom_prompt_bank['Time-Aware_Belief_Solver']
        character_prompt = ''
        if len(sorted_char_lst) == 2:
            character_prompt = f'{sorted_char_lst[0]} and {sorted_char_lst[1]}'
        else:
            for idx, char in enumerate(sorted_char_lst):

                if idx == len(sorted_char_lst) - 1:
                    character_prompt += f'and {char}'
                else:
                    character_prompt += f'{char}, '

        character_tbsc = character_tbsc_dict[sorted_char_lst[-1]]

        # final stage prompt
        cur_prompt = solver_prompt_lst[2]
        cur_prompt = cur_prompt.replace('{{perspective}}', narrative_with_temporal_space) \
            .replace('{{character}}', sorted_char_lst[-1]) \
            .replace('{{character list}}', character_prompt) \
            .replace('{{question}}', reduced_question) \
            .replace('{{common belief}}', common_narrative) \
            .replace('{{answer1}}', temp_answer1) \
            .replace('{{answer2}}', temp_answer2) \
            .replace('{{character belief}}', character_tbsc) \
            .replace('{{character1}}', sorted_char_lst[0]) \
            .replace('{{character2}}', sorted_char_lst[1]) 

        if len(sorted_char_lst) > 2:
            cur_prompt = cur_prompt.replace('{{character3}}', sorted_char_lst[2])
        else:
            cur_prompt = cur_prompt.replace('{{character3}}', 'others')

        cur_msg = [{
            'role': 'user',
            'content': cur_prompt,
        }]

        return cur_msg

    
    async def __belief_solver(
        self,
        high_order_qa_msg_dict: dict,
    ):
        stage1_prompt_lst = []
        stage2_prompt_lst = []
        stage3_prompt_lst = []
        for key, val in high_order_qa_msg_dict.items():
            stage1_prompt_lst.append(val['msg'][0])
            stage2_prompt_lst.append(val['msg'][1])
            stage3_prompt_lst.append(val['msg'][2])

        stage1_results = [
            self.async_model.inference(
                model=self.model_name,
                message=stage1_prompt,
                temperature=0.,
            ) for stage1_prompt in stage1_prompt_lst
        ]
        stage1_results = await atqdm.gather(*stage1_results, desc='Belief Solver Stage 1')

        stage2_results = [
            self.async_model.inference(
                model=self.model_name,
                message=stage2_prompt,
                temperature=0.,
            ) for stage2_prompt in stage2_prompt_lst
        ]
        stage2_results = await atqdm.gather(*stage2_results, desc='Belief Solver Stage 2')
        
        for idx, (answer1, answer2, prompt3) in enumerate(zip(stage1_results, stage2_results, stage3_prompt_lst)):
            prompt3[-1]['content'] = prompt3[-1]['content'] \
                .replace('{{answer1}}', answer1) \
                .replace('{{answer2}}', answer2)

            stage3_prompt_lst[idx] = prompt3

        high_order_qa_result_lst = [
            self.async_model.inference(
                model=self.model_name,
                message=stage3_prompt,
                temperature=0.,
            ) for stage3_prompt in stage3_prompt_lst
        ]
        high_order_qa_result_lst = await atqdm.gather(*high_order_qa_result_lst, desc='Belief Solver Stage 3')

        high_order_qa_result_dict = {
            key: {
                'response': val, 
                'correct_letter': cur_msg['correct_letter'],
                'prompt': [
                    stage1_prompt_lst[idx],
                    stage2_prompt_lst[idx],
                    stage3_prompt_lst[idx],
                ],
            }
            for idx, (key, val, cur_msg) in enumerate(zip(
                high_order_qa_msg_dict.keys(), 
                high_order_qa_result_lst,
                high_order_qa_msg_dict.values()
        ))}

        return high_order_qa_result_dict


    def __high_order_qa_msg(
        self, 
        timetom_prompt_bank: dict, 
        narrative_with_temporal_space: str, 
        sorted_char_lst: list,
        question: str,
        reduced_question: str,
        character_tbsc_dict: dict,
        belief_solver: bool,
        question_type: str = '',
    ) -> list:

        narrative_time_stamps = self.__extract_time_stamps(
            narrative_with_temporal_space, 
            return_narrative=True
        )

        all_character_beliefs = ''
        for character, tbsc in character_tbsc_dict.items():
            all_character_beliefs += f'{character}:\n{tbsc}\n\n'
        
        char_time_stamp_dict = {char: [] for char in sorted_char_lst}
        for char in sorted_char_lst:
            character_narrative = character_tbsc_dict[char]
            character_time_stamps = self.__extract_time_stamps(character_narrative)
            char_time_stamp_dict[char] = character_time_stamps

        # TBSC solver for high-order ToM
        common_time_stamps = set.intersection(*map(set, char_time_stamp_dict.values()))
        common_time_stamps = sorted(list(common_time_stamps))
        common_narrative = '\n'.join([ele[1] for ele in narrative_time_stamps if ele[0] in common_time_stamps])

        if belief_solver:
            qa_prompt1 = self.__belief_solver_stage1_msg(
                timetom_prompt_bank=timetom_prompt_bank,
                narrative_with_temporal_space=narrative_with_temporal_space,
                character_tbsc_dict=character_tbsc_dict,
                sorted_char_lst=sorted_char_lst,
                question=question,
                reduced_question=reduced_question,
            )
            qa_prompt2 = self.__belief_solver_stage2_msg(
                timetom_prompt_bank=timetom_prompt_bank,
                common_narrative=common_narrative,
                sorted_char_lst=sorted_char_lst,
                question=question,
                reduced_question=reduced_question,
            )
            qa_prompt3 = self.__belief_solver_stage3_msg(
                timetom_prompt_bank=timetom_prompt_bank,
                narrative_with_temporal_space=narrative_with_temporal_space,
                common_narrative=common_narrative,
                sorted_char_lst=sorted_char_lst,
                question=question,
                reduced_question=reduced_question,
                character_tbsc_dict=character_tbsc_dict,
            )

            qa_msg = [qa_prompt1, qa_prompt2, qa_prompt3]

        else:
            # QA prompt for Dialogue (FANToM)
            fact_info = ''
            if 'Information:' in reduced_question:
                fact_info, reduced_question = reduced_question.strip().rsplit('\n', 1)
                fact_info = fact_info.replace('Information:', '').strip()
            elif 'Question:' in reduced_question:
                fact_info, reduced_question = reduced_question.split('\n')
                fact_info = fact_info.replace('Question:', '').strip()

            if question_type == 'answerability_binary':
                qa_prompt = timetom_prompt_bank['Time-Aware_Answerability_Binary_Question_Answer']
            elif question_type == 'infoaccess_binary':
                qa_prompt = timetom_prompt_bank['Time-Aware_Infoaccess_Binary_Question_Answer']
            elif question_type == 'answerability_list':
                qa_prompt = timetom_prompt_bank['Time-Aware_Answerability_List_Question_Answer']
            elif question_type == 'infoaccess_list':
                qa_prompt = timetom_prompt_bank['Time-Aware_Infoaccess_List_Question_Answer']
            else:
                qa_prompt = timetom_prompt_bank['Time-Aware_Belief_QA_no_Compression']

            qa_prompt = qa_prompt.replace('{{compressed perspective}}', common_narrative) \
                .replace('{{perspective}}', common_narrative) \
                .replace('{{character}}', sorted_char_lst[-1]) \
                .replace('{{question}}', reduced_question) \
                .replace('{{all character beliefs}}', all_character_beliefs) \
                .replace('{{fact info}}', fact_info)

            qa_msg = [{
                'role': 'user',
                'content': qa_prompt,
            }]

        return qa_msg


    @staticmethod
    def __classify_tom_order(question: str, char_lst: list) -> tuple:
        order = sum([char in question for char in char_lst])

        char_order = [[question.index(char), char] for char in char_lst if char in question]
        sorted_char_order = sorted(char_order, key=lambda x: x[0])
        sorted_char_order = [ele[1] for ele in sorted_char_order]

        return order, sorted_char_order

    
    async def evaluate(
        self, 
        data: dict, 
        prompt_bank: dict, 
        timetom_prompt_bank: dict, 
        char_name_dict: dict, 
        eval_config: dict,
        data_name: str,
        mc_probing: bool,
        seed: int,
    ):

        candidate_keys = eval_config['eval_keys']
        candidate_delim = eval_config['delimiter']

        #===============================================================================================================

        # Module: construct temporal space
        temporal_space_msg_lst = []
        temporal_space_key_lst = []
        for key, val in data.items():
            narrative = val['narrative']
            cur_msg = self.__construct_temporal_space_msg(timetom_prompt_bank, narrative)
            temporal_space_msg_lst.append(cur_msg)
            temporal_space_key_lst.append(key)

        if self.to_jsonl:

            processed_cache_fpath = (
                f'../data/openai_batch_exp_processed/{self.model_name}' 
                f'/{data_name}/{seed}/tomi_timetom-part1.jsonl'
            )
            if os.path.exists(processed_cache_fpath):

                narrative_with_temporal_space_lst = self.file_io.load_jsonl(processed_cache_fpath)
                narrative_with_temporal_space_lst = [
                    ele['response']['body']['choices'][0]['message']['content'] 
                    for ele in narrative_with_temporal_space_lst
                ]

            else:

                out_jsonl = self.convert_msg_to_jsonl(
                    msg_lst=temporal_space_msg_lst,
                    key_lst=temporal_space_key_lst,
                )

                cache_fpath = (
                    f'../data/openai_batch_exp_data/{self.model_name}/' 
                    f'{data_name}/{seed}/timetom_part1_seed{seed}.jsonl'
                )
                self.file_io.save_jsonl(
                    obj=out_jsonl,
                    fpath=cache_fpath,
                )
                print((
                    f'TimeToM Part1 Not Found, use JSONL saved at\n{cache_fpath}\n' 
                    'to continue with OpenAI Batch Inference'
                ))
                raise SystemExit()
        else:
            narrative_with_temporal_space_lst = [
                self.async_model.inference(
                    model=self.model_name,
                    message=temporal_space_msg,
                    temperature=0.,
                ) for temporal_space_msg in temporal_space_msg_lst
            ]
            narrative_with_temporal_space_lst = await atqdm.gather(
                *narrative_with_temporal_space_lst, 
                desc='Constructing Temporal Space'
            )

        narrative_with_temporal_space_dict = {}
        for idx, (key, val) in enumerate(data.items()):
            narrative_with_temporal_space_dict[key] = narrative_with_temporal_space_lst[idx]

        #===============================================================================================================

        # Module: construct TBSC
        tbsc_msg_lst = []
        tbsc_key_lst = []
        for key, val in data.items():
            char_lst = char_name_dict[key]
            narrative = val['narrative']
            question_lst = val['questions']

            containers = []
            if 'plot_info' in val.keys():
                plot_info = val['plot_info']
                containers = [plot_info['original_place'], plot_info['move_to_place']]

            relevant_chars = [
                char for char in char_lst if char in ' '.join([ele['question'] for ele in question_lst])
            ]

            narrative_with_temporal_space = narrative_with_temporal_space_dict[key]

            char_narrative_dict = {char: '' for char in relevant_chars}
            if relevant_chars:
                for char in relevant_chars:
                    cur_msg = self.__construct_TBSC_msg(
                        timetom_prompt_bank, 
                        narrative_with_temporal_space, 
                        char,
                    )
                    tbsc_msg_lst.append((key, char, cur_msg))
                    tbsc_key_lst.append(key)

        if self.to_jsonl:
            processed_cache_fpath = (
                f'../data/openai_batch_exp_processed/{self.model_name}' 
                f'/{data_name}/{seed}/tomi_timetom-part2.jsonl'
            )
            if os.path.exists(processed_cache_fpath):

                character_tbsc_lst = self.file_io.load_jsonl(processed_cache_fpath)
                character_tbsc_lst = [
                    ele['response']['body']['choices'][0]['message']['content'] 
                    for ele in character_tbsc_lst
                ]

            else:
                
                tbsc_msg_only_lst = [ele[-1] for ele in tbsc_msg_lst]
                out_jsonl = self.convert_msg_to_jsonl(
                    msg_lst=tbsc_msg_only_lst,
                    key_lst=tbsc_key_lst,
                )

                cache_fpath = (
                    f'../data/openai_batch_exp_data/{self.model_name}/' 
                    f'{data_name}/{seed}/timetom_part2_seed{seed}.jsonl'
                )
                self.file_io.save_jsonl(
                    obj=out_jsonl,
                    fpath=cache_fpath,
                )
                print((
                    f'TimeToM Part2 Not Found, use JSONL saved at\n{cache_fpath}\n' 
                    'to continue with OpenAI Batch Inference'
                ))
                raise SystemExit()
        else:
            character_tbsc_lst = [
                self.async_model.inference(
                    model=self.model_name,
                    message=tbsc_prompt[-1],
                    temperature=0.,
                ) for tbsc_prompt in tbsc_msg_lst
            ]

            character_tbsc_lst = await atqdm.gather(*character_tbsc_lst, desc='Constructing TBSC')

        character_tbsc_dict = {}
        for tbsc_response, tbsc_msg_tup in zip(character_tbsc_lst, tbsc_msg_lst):
            key, char, _ = tbsc_msg_tup
            if key not in character_tbsc_dict:
                character_tbsc_dict[key] = {}

            character_tbsc_dict[key][char] = tbsc_response

        #===============================================================================================================

        # Module: Belief Compression for First-Order ToM, Only needed for narrative datasets
        if data_name != 'fantom':
            belief_compression_msg_dict = {}
            for key, val in data.items():
                char_lst = char_name_dict[key]
                question_lst = val['questions']
                narrative_with_temporal_space = narrative_with_temporal_space_dict[key]

                relevant_chars = [
                    char for char in char_lst if char in ' '.join([ele['question'] for ele in question_lst])
                ]

                for idx, q_dict in enumerate(question_lst):
                    cur_question = q_dict['question']

                    if relevant_chars:
                        tom_order, sorted_char_order = self.__classify_tom_order(cur_question, relevant_chars)
                    else:
                        tom_order = 0

                    if tom_order == 1:
                        pivot_char = sorted_char_order[0]
                        char_tbsc = character_tbsc_dict[key][pivot_char]
                        cur_compression_msg = self.__belief_compression(
                            timetom_prompt_bank=timetom_prompt_bank, 
                            character_tbsc=char_tbsc, 
                            character=relevant_chars[0]
                        )

                        belief_compression_msg_dict[(key, idx)] = cur_compression_msg

            belief_compression_msg_lst = [
                self.async_model.inference(
                    model=self.model_name,
                    message=cur_msg,
                    temperature=0.,
                ) for cur_msg in list(belief_compression_msg_dict.values())
            ]
            belief_compression_msg_lst = await atqdm.gather(
                *belief_compression_msg_lst, desc='Compression Belief for First-Order ToM'
            )
            belief_compression_dict = {}
            for key in belief_compression_msg_dict.keys():
                belief_compression_dict[key] = belief_compression_msg_lst.pop(0)
        else:
            belief_compression_dict = {}

        #===============================================================================================================

        # Module: QA for First-Order and High-Order ToM
        first_order_qa_msg_dict, high_order_qa_msg_dict = {}, {}
        for key, val in data.items():
            char_lst = char_name_dict[key]
            question_lst = val['questions']
            narrative_with_temporal_space = narrative_with_temporal_space_dict[key]

            containers = []
            if 'plot_info' in val.keys():
                plot_info = val['plot_info']
                containers = [plot_info['original_place'], plot_info['move_to_place']]

            relevant_chars = [
                char for char in char_lst if char in ' '.join([ele['question'] for ele in question_lst])
            ]

            for idx, q_dict in enumerate(question_lst):
                cur_question = q_dict['question']

                if relevant_chars:
                    tom_order, sorted_char_order = self.__classify_tom_order(cur_question, relevant_chars)
                else:
                    tom_order = 0

                if belief_compression_dict:
                    question_type = ''
                else:
                    correct_answer = q_dict['correct_answer']
                    q_type = q_dict['question_type']
                    if correct_answer in ['yes', 'no']:
                        if 'answerability' in q_type:
                            question_type = 'answerability_binary'
                        elif 'access' in q_type:
                            question_type = 'infoaccess_binary'
                    elif isinstance(correct_answer, list):
                        if 'answerability' in q_type:
                            question_type = 'answerability_list'
                        elif 'access' in q_type:
                            question_type = 'infoaccess_list'
                    else:
                        question_type = 'mc'

                    # adjust for FANToM
                    mc_probing = True if question_type == 'mc' else False

                question_prompt, reduced_question_prompt, correct_letter = self.__make_question_prompts(
                    question=q_dict,
                    candidate_keys=candidate_keys,
                    candidate_delim=candidate_delim,
                    prompt_bank=prompt_bank,
                    containers=containers,
                    mc_probing=mc_probing,
                )

                # some narratives do not have any ToM question, hence there will be no TBSC
                cur_character_tbsc_dict = character_tbsc_dict.get(key, {})
                if tom_order == 1:

                    if belief_compression_dict:
                        compressed_narrative = belief_compression_dict[(key, idx)]
                    else:
                        compressed_narrative = character_tbsc_dict[key][sorted_char_order[-1]]

                    cur_msg = self.__first_order_qa_msg(
                        timetom_prompt_bank=timetom_prompt_bank, 
                        compressed_narrative=compressed_narrative, 
                        character=sorted_char_order[-1], 
                        question_prompt=question_prompt,
                        question_type=question_type,
                        character_tbsc_dict=cur_character_tbsc_dict,
                    )
                    first_order_qa_msg_dict[(key, idx)] = {
                        'msg': cur_msg,
                        'correct_letter': correct_letter,
                    }

                elif tom_order > 1:

                    # NOTE: make prompts for high-order ToM QA with belief solver
                    cur_msg = self.__high_order_qa_msg(
                        timetom_prompt_bank=timetom_prompt_bank, 
                        narrative_with_temporal_space=narrative_with_temporal_space, 
                        sorted_char_lst=sorted_char_order,
                        question=question_prompt,
                        reduced_question=reduced_question_prompt,
                        character_tbsc_dict=cur_character_tbsc_dict,
                        belief_solver=self.use_belief_solver,
                        question_type=question_type,
                    )
                    high_order_qa_msg_dict[(key, idx)] = {
                        'msg': cur_msg,
                        'correct_letter': correct_letter,
                    }

                else:
                    cur_msg = self.__first_order_qa_msg(
                        timetom_prompt_bank, 
                        narrative_with_temporal_space, 
                        'omniscent-view', 
                        question_prompt,
                        character_tbsc_dict=cur_character_tbsc_dict,
                    )
                    first_order_qa_msg_dict[(key, idx)] = {
                        'msg': cur_msg,
                        'correct_letter': correct_letter,
                    }

        first_order_qa_result_lst = [
            self.async_model.inference(
                model=self.model_name,
                message=cur_msg['msg'],
                temperature=0.,
            ) for cur_msg in list(first_order_qa_msg_dict.values())
        ]
        first_order_qa_result_lst = await atqdm.gather(
            *first_order_qa_result_lst, desc='Evaluating First-Order QA'
        )
        first_order_qa_result_dict = {
            key: {
                'response': val, 
                'correct_letter': cur_msg['correct_letter'],
                'prompt': cur_msg['msg'][-1]['content'],
            }
            for key, val, cur_msg in zip(
                first_order_qa_msg_dict.keys(), 
                first_order_qa_result_lst, 
                first_order_qa_msg_dict.values()
        )}

        if self.use_belief_solver:
            high_order_qa_result_dict = await self.__belief_solver(
                high_order_qa_msg_dict=high_order_qa_msg_dict,
            )
        else:
            high_order_qa_result_lst = [
                self.async_model.inference(
                    model=self.model_name,
                    message=cur_msg['msg'],
                    temperature=0.,
                ) for cur_msg in list(high_order_qa_msg_dict.values())
            ]
            high_order_qa_result_lst = await atqdm.gather(
                *high_order_qa_result_lst, desc='Evaluating High-Order QA'
            )
            high_order_qa_result_dict = {
                key: {
                    'response': val, 
                    'correct_letter': cur_msg['correct_letter'],
                    'prompt': cur_msg['msg'][-1]['content'],
                }
                for key, val, cur_msg in zip(
                    high_order_qa_msg_dict.keys(), 
                    high_order_qa_result_lst,
                    high_order_qa_msg_dict.values()
            )}

        for key, val in data.items():
            question_lst = val['questions']
            for idx, q_dict in enumerate(question_lst):

                if (key, idx) in first_order_qa_result_dict:
                    result, correct_letter, cur_prompt = first_order_qa_result_dict[(key, idx)].values()
                    data[key]['questions'][idx]['prompt'] = cur_prompt
                    data[key]['questions'][idx]['predicted'] = result
                elif (key, idx) in high_order_qa_result_dict:
                    result, correct_letter, cur_prompt = high_order_qa_result_dict[(key, idx)].values()
                    data[key]['questions'][idx]['prompt'] = cur_prompt
                    data[key]['questions'][idx]['predicted'] = result
                else:
                    raise ValueError(f'Key {key} and Index {idx} not found in results')

                if correct_letter:
                    data[key]['questions'][idx]['correct_letter'] = correct_letter
    
        return data


    @staticmethod
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


async def run_timetom(
    evaluator: TimeToMEvaluator,
    data_path: str,
    model: str,
    prompt_path: str,
    timetom_prompt_path: str,
    belief_solver: bool,
    seed: int,
    data_name: str,
    mc_probing: bool = False,
):
    file_io = FileIO()

    data = file_io.load_json(data_path)
    data_keys = list(data.keys())

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

    prompt_bank = file_io.load_yaml(prompt_path)
    timetom_prompt_bank = file_io.load_yaml(timetom_prompt_path)

    if 'fantom' in data_name:
        timetom_prompt_bank = timetom_prompt_bank['Dialogue']
        SAMPLE_SIZE = 50
    else:
        timetom_prompt_bank = timetom_prompt_bank['Narrative']
        SAMPLE_SIZE = 100

    char_names = file_io.load_json(f'../data/masktom/ner/{data_name}_char_name_seed[{seed}].json')
    eval_config = file_io.load_yaml('./configs/dataset_eval_config.yml')
    eval_config = eval_config[data_name]
    prompt_bank = prompt_bank[eval_config['eval_method']]

    # limit eval size to 30 at the moment for prelim exps
    random.seed(seed)
    np.random.seed(seed)
    sampled_keys = np.random.choice(data_keys, SAMPLE_SIZE, replace=False)
    sampled_data = {key: val for (key, val) in data.items() if key in sampled_keys}

    result = await evaluator.evaluate(
        data=sampled_data, 
        prompt_bank=prompt_bank,
        timetom_prompt_bank=timetom_prompt_bank, 
        char_name_dict=char_names, 
        eval_config=eval_config,
        data_name=data_name,
        mc_probing=mc_probing,
        seed=seed,
    )

    if belief_solver:
        postfix = '_belief-solver'
    else:
        postfix = ''

    file_io.save_json(result, f'../data/prelim_results/{data_name}/{model}/{seed}/{data_name}{postfix}_timetom.json')
