import os
import re
import string
import random
import numpy as np
from tqdm import tqdm
from components.utils import FileIO
from components.llms import OpenAIInference


class TimeToMEvaluator:

    file_io = FileIO()
    AVAILABLE_MODELS = file_io.load_yaml('./configs/model_info.yml')

    def __init__(
        self, 
        model_name: str, 
        use_blief_solver: bool, 
        opentom_flag: bool,
        mc_probing: bool = False,
    ) -> None:

        assert model_name in self.AVAILABLE_MODELS.keys(), \
        f'{model_name} not in {self.AVAILABLE_MODELS.keys()}'
        model_config_tup = self.AVAILABLE_MODELS[model_name]
        model_class, model_path, model_config, model_quantization = model_config_tup

        vllm_config = self.file_io.load_yaml('./configs/vllm_configs.yml')
        cur_model_config = vllm_config[model_name]

        self.use_belief_solver = use_blief_solver
        self.opentom_flag = opentom_flag
        self.opentom_flag = opentom_flag
        self.model_name = model_name
        self.mc_probing = mc_probing
        self.probing_method = 'freeform_generation' if mc_probing else 'multiple_choice'

        # if model_class == 'hf':
        #     # self.model = HFInference()
        #
        #     # quantization = model_quantization if model_quantization else 32
        #
        #     # # WARNING: uncomment this when running eval 
        #     # self.model.init_model(
        #     #     model_path, 
        #     #     model_name, 
        #     #     quantization=quantization,
        #     #     config_path=model_config
        #     # )
        #
        #     self.model = vLLMInference(model_name=model_name)
        #
        # elif model_class == 'gpt':
        #     ...

        self.model = OpenAIInference(
            base_url=cur_model_config['base_url'],
            api_key=cur_model_config['api_key'],
        )


    def __construct_temporal_space(self, timetom_prompt_bank: dict, narrative: str) -> str:
        """
        Construct the temporal space of the narrative. 
        Section 2.1
        """

        temporal_state_prompt = timetom_prompt_bank['Constructing_Temporal_Space'] \
            .replace('{{story}}', narrative)

        temporal_state_prompt = [{
            'role': 'user',
            'content': temporal_state_prompt,
        }]

        narrative_with_temporal_space = self.model.inference(
            model=self.model_name,
            message=temporal_state_prompt,
            temperature=0.,
        )

        return narrative_with_temporal_space

    def __construct_TBSC(
        self, 
        timetom_prompt_bank: dict, 
        narrative_with_temporal_space: str, 
        character: str
    ) -> str:
        """
        Construct the TBSC of the narrative. 
        Section 2.2
        """

        tbsc_prompt = timetom_prompt_bank['Temporal_Belief_State_Chain_Construction'] \
            .replace('{{character}}', character) \
            .replace('{{story}}', narrative_with_temporal_space)

        tbsc_prompt = [{
            'role': 'user',
            'content': tbsc_prompt,
        }]
        character_tbsc = self.model.inference(
            model=self.model_name,
            message=tbsc_prompt,
            temperature=0.,
        )

        return character_tbsc


    def __belief_compression(
        self, 
        timetom_prompt_bank: dict, 
        character_tbsc: str, 
        character: str
    ) -> str:
        """
        Belief compression of the narrative. 
        Section 2.3
        """

        belief_compression_prompt = timetom_prompt_bank['Belief_Compression'] \
            .replace('{{perspective}}', character_tbsc) \
            .replace('{{character}}', character)

        belief_compression_prompt = [{
            'role': 'user',
            'content': belief_compression_prompt,
        }]
        compressed_narrative = self.model.inference(
            model=self.model_name,
            message=belief_compression_prompt,
            temperature=0.,
        )

        return compressed_narrative


    def __make_question_prompts(
        self, 
        question: dict, 
        candidate_keys: list, 
        candidate_delim: str,
        prompt_bank: dict,
        containers: list,
        mc_probing: bool
    ) -> tuple:

        if self.opentom_flag:

            candidate_answers = prompt_bank['candidate_answers']

            cur_question = question['question']
            candidate_lst = []

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

            candidate_lst = [str(ele) for ele in candidate_lst]

            candidate_str = cur_question + f' The candidate answers are: {", ".join(candidate_lst)}.'

            reduced_candidate_str = question['reduced_question'] + \
                f' Choose from one of the following options: {", ".join(candidate_lst)}.'
            correct_letter = ''

        else:
            if candidate_delim:
                candidate_lst = question[candidate_keys[0]].split(candidate_delim)
                candidate_lst = [ele.split('.', 1)[-1].strip() for ele in candidate_lst]
            else:
                candidate_lst = [question[ele] for ele in candidate_keys]

            random.shuffle(candidate_lst)
            correct_ans = question[candidate_keys[1]]
            correct_idx = candidate_lst.index(correct_ans)

            if mc_probing:
                candidate_letters = list(string.ascii_uppercase[:len(candidate_lst)])
                correct_letter = candidate_letters[correct_idx]
                candidate_str = '\n' + '\n'.join([
                    f'{letter}. {ans}' for letter, ans in zip(candidate_letters, candidate_lst)
                ])
            else:
                candidate_str = str(candidate_lst)

            candidate_str = question['question'] + \
                ' Choose from one of the following options ' + \
                candidate_str

            reduced_candidate_str = question['reduced_question'] + '\n' + candidate_str

            correct_letter = string.ascii_uppercase[correct_idx]

        return candidate_str, reduced_candidate_str, correct_letter


    def __first_order_qa(
        self, 
        timetom_prompt_bank: dict, 
        compressed_narrative: str, 
        character: str, 
        question_prompt: str
    ) -> tuple:

        qa_prompt = timetom_prompt_bank['Time-Aware_Belief_QA']
        qa_prompt = qa_prompt.replace('{{compressed perspective}}', compressed_narrative) \
            .replace('{{character}}', character) \
            .replace('{{question}}', question_prompt)

        qa_msg = [{
            'role': 'user',
            'content': qa_prompt.strip(),
        }]

        result = self.model.inference(
            model=self.model_name,
            message=qa_msg,
            temperature=0.,
        )

        return result, qa_prompt


    @staticmethod
    def __extract_time_stamps(
        narrative_with_temporal_space: str, 
        return_narrative: bool = False
    ) -> list:
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
    

    def __belief_solver(
        self, 
        timetom_prompt_bank: dict, 
        narrative_with_temporal_space: str, 
        common_narrative: str,
        sorted_char_lst: list,
        question: str,
        reduced_question: str,
    ) -> tuple:

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

        # stage 1 prompt
        cur_prompt = solver_prompt_lst[0]
        cur_prompt = cur_prompt.replace('{{perspective}}', narrative_with_temporal_space) \
            .replace('{{character}}', sorted_char_lst[0]) \
            .replace('{{question}}', reduced_question)
        
        cur_msg = [{
            'role': 'user',
            'content': cur_prompt,
        }]

        temp_answer1 = self.model.inference(
            model=self.model_name,
            message=cur_msg,
            temperature=0.,
        )

        # stage 2 prompt
        cur_prompt = solver_prompt_lst[1]
        cur_prompt = cur_prompt.replace('{{character list}}', character_prompt) \
            .replace('{{common_belief}}', common_narrative) \
            .replace('{{question}}', reduced_question)

        cur_msg = [{
            'role': 'user',
            'content': cur_prompt,
        }]

        temp_answer2 = self.model.inference(
            model=self.model_name,
            message=cur_msg,
            temperature=0.,
        )

        # final stage prompt
        cur_prompt = solver_prompt_lst[2]
        cur_prompt = cur_prompt.replace('{{perspective}}', narrative_with_temporal_space) \
            .replace('{{character}}', sorted_char_lst[0]) \
            .replace('{{character list}}', character_prompt) \
            .replace('{{question}}', reduced_question) \
            .replace('{{common_belief}}', common_narrative) \
            .replace('{{answer1}}', temp_answer1) \
            .replace('{{answer2}}', temp_answer2)

        cur_msg = [{
            'role': 'user',
            'content': cur_prompt,
        }]

        result = self.model.inference(
            model=self.model_name,
            message=cur_msg,
            temperature=0.,
        )

        return result, cur_prompt


    def __high_order_qa(
        self, 
        timetom_prompt_bank: dict, 
        narrative_with_temporal_space: str, 
        sorted_char_lst: list,
        question: str,
        reduced_question: str,
    ) -> tuple:

        narrative_time_stamps = self.__extract_time_stamps(
            narrative_with_temporal_space, 
            return_narrative=True
        )
        
        char_time_stamp_dict = {char: [] for char in sorted_char_lst}
        for char in sorted_char_lst:
            character_narrative = self.__construct_TBSC(
                timetom_prompt_bank, 
                narrative_with_temporal_space, 
                char
            )
            character_time_stamps = self.__extract_time_stamps(character_narrative)
            char_time_stamp_dict[char] = character_time_stamps

        # TBSC solver for high-order ToM
        common_time_stamps = set.intersection(*map(set, char_time_stamp_dict.values()))
        common_time_stamps = sorted(list(common_time_stamps))
        common_narrative = '\n'.join([
            ele[1] for ele in narrative_time_stamps if ele[0] in common_time_stamps
        ])

        if self.use_belief_solver:
            result, qa_prompt = self.__belief_solver(
                timetom_prompt_bank, 
                narrative_with_temporal_space, 
                common_narrative, 
                sorted_char_lst, 
                question,
                reduced_question,
            )
        else:
            qa_prompt = timetom_prompt_bank['Time-Aware_Belief_QA_no_Compression']
            qa_prompt = qa_prompt.replace('{{perspective}}', common_narrative) \
                .replace('{{character}}', sorted_char_lst[-1]) \
                .replace('{{question}}', reduced_question)

            qa_msg = [{
                'role': 'user',
                'content': qa_prompt,
            }]

            result = self.model.inference(
                model=self.model_name,
                message=qa_msg,
                temperature=0.,
            )

        return result, qa_prompt


    @staticmethod
    def __classify_tom_order(question: str, char_lst: list) -> tuple:
        order = sum([char in question for char in char_lst])

        char_order = [[question.index(char), char] for char in char_lst if char in question]
        sorted_char_order = sorted(char_order, key=lambda x: x[0])
        sorted_char_order = [ele[1] for ele in sorted_char_order]

        return order, sorted_char_order

    
    def evaluate(
        self, 
        data: dict, 
        prompt_bank: dict, 
        timetom_prompt_bank: dict, 
        char_name_dict: dict, 
        eval_config: dict
    ):

        candidate_keys = eval_config['eval_keys']
        candidate_delim = eval_config['delimiter']

        # construct temporal space
        for key, val in tqdm(data.items(), desc='Evaluating with TimeToM'):
            char_lst = char_name_dict[key]
            narrative = val['narrative']
            question_lst = val['questions']
            narrative_with_temporal_space = self.__construct_temporal_space(
                timetom_prompt_bank, 
                narrative
            )

            containers = []
            if 'plot_info' in val.keys():
                plot_info = val['plot_info']
                containers = [plot_info['original_place'], plot_info['move_to_place']]

            relevant_chars = [
                char for char in char_lst 
                if char in ' '.join([ele['question'] for ele in question_lst])
            ]

            char_narrative_dict = {char: '' for char in relevant_chars}
            if relevant_chars:
                for char in relevant_chars:
                    char_narrative = self.__construct_TBSC(
                        timetom_prompt_bank, 
                        narrative_with_temporal_space, 
                        char,
                    )
                    char_narrative_dict[char] = char_narrative

            for idx, q_dict in enumerate(question_lst):
                cur_question = q_dict['question']

                if relevant_chars:
                    tom_order, sorted_char_order = self.__classify_tom_order(
                        cur_question, 
                        relevant_chars
                    )
                else:
                    tom_order = 0

                if tom_order == 1:
                    compressed_narrative = self.__belief_compression(
                        timetom_prompt_bank, 
                        narrative_with_temporal_space, 
                        relevant_chars[0]
                    )

                    question_prompt, _, correct_letter = self.__make_question_prompts(
                        question=q_dict,
                        candidate_keys=candidate_keys,
                        candidate_delim=candidate_delim,
                        prompt_bank=prompt_bank,
                        containers=containers,
                        mc_probing=self.mc_probing,
                    )

                    result, eval_prompt = self.__first_order_qa(
                        timetom_prompt_bank, 
                        compressed_narrative, 
                        relevant_chars[0], 
                        question_prompt
                    )

                elif tom_order > 1:

                    question_prompts = self.__make_question_prompts(
                        question=q_dict,
                        candidate_keys=candidate_keys,
                        candidate_delim=candidate_delim,
                        prompt_bank=prompt_bank,
                        containers=containers,
                        mc_probing=self.mc_probing,
                    )
                    question_prompt, reduced_question_prompt, correct_letter = question_prompts

                    result, eval_prompt = self.__high_order_qa(
                        timetom_prompt_bank, 
                        narrative_with_temporal_space, 
                        sorted_char_order,
                        question_prompt,
                        reduced_question_prompt,
                    )

                else:
                    question_prompt, _, correct_letter = self.__make_question_prompts(
                        question=q_dict,
                        candidate_keys=candidate_keys,
                        candidate_delim=candidate_delim,
                        prompt_bank=prompt_bank,
                        containers=containers,
                        mc_probing=self.mc_probing,
                    )

                    result, eval_prompt = self.__first_order_qa(
                        timetom_prompt_bank, 
                        narrative_with_temporal_space, 
                        'omniscent-view', 
                        question_prompt
                    )

                data[key]['questions'][idx]['predicted'] = result
                data[key]['questions'][idx]['eval_prompt'] = eval_prompt

                if correct_letter:
                    data[key]['questions'][idx]['correct_letter'] = correct_letter
    
        return data


def run_timetom(
    evaluator: TimeToMEvaluator,
    data_path: str,
    model: str,
    prompt_path: str,
    timetom_prompt_path: str,
    belief_solver: bool,
    seed: int,
    mc_probing: bool,
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

    prompt_bank = file_io.load_yaml(prompt_path)
    timetom_prompt_bank = file_io.load_yaml(timetom_prompt_path)

    char_names = file_io.load_json(f'../data/masktom/ner/{data_name}_char_name_seed[{seed}].json')
    eval_config = file_io.load_yaml('./configs/dataset_eval_config.yml')
    eval_config = eval_config[data_name]
    prompt_bank = prompt_bank[eval_config['eval_method']]

    # limit eval size to 30 at the moment for prelim exps
    SAMPLE_SIZE = 100
    random.seed(seed)
    np.random.seed(seed)
    sampled_keys = np.random.choice(data_keys, SAMPLE_SIZE, replace=False)
    sampled_data = {key: val for (key, val) in data.items() if key in sampled_keys}

    result = evaluator.evaluate(
        data=sampled_data, 
        prompt_bank=prompt_bank,
        timetom_prompt_bank=timetom_prompt_bank, 
        char_name_dict=char_names, 
        eval_config=eval_config,
    )

    postfix = ''
    if belief_solver:
        postfix = '_belief-solver'
    if mc_probing:
        postfix += '_mc-probing'

    file_io.save_json(
        result, 
        (
            f'../data/prelim_results/{data_name}/{model}/{seed}/' 
            f'{data_name}{postfix}_timetom.json'
    ))