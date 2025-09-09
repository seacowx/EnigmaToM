from copy import deepcopy
from tqdm.asyncio import tqdm
from components.llms import OpenAIInference
from components.utils import FileIO, load_model


def build_few_shot_template(prompt_bank: dict, data_name: str):
    instruction_template = prompt_bank['reduce_tom_question']
    example_dict = prompt_bank[data_name]

    example_question_lst = example_dict['tom_reduction_questions'].values()
    example_response_lst = example_dict['tom_reduction_responses'].values()

    example_instruction_lst = []
    for ques, resp in zip(example_question_lst, example_response_lst):
        example_instruction_lst.extend([
            {
                'role': 'user',
                'content': instruction_template.replace('{{tom question}}', ques),
            },
            {
                'role': 'assistant',
                'content': f'Answer: {resp}',
            }
        ])
        
    return example_instruction_lst


async def reduce_tom_questions(
    data: dict,
    data_name: str,
    model_name: str,
    prompt_path: str,
    data_character_dict: dict
) -> dict:

    file_io = FileIO()

    model = load_model(
        model_name=model_name,
        no_async=False,
    )

    prompt_bank = file_io.load_yaml(prompt_path)

    few_shot_template = build_few_shot_template(prompt_bank, data_name)
    instruction_template = prompt_bank['reduce_tom_question']

    # Question types that should not be reduced:
    # 1. All questions in BigToM
    # 2. Attitude and multihop questions in OpenToM
    msg_lst = []
    msg_idx = 0
    msg_idx_to_question = {}
    for key, val in data.items():
        question_lst = val['questions']
        character_lst = data_character_dict[key]

        for idx, question in enumerate(tqdm(question_lst, position=1, leave=False, desc=f'Processing {key}')):

            cur_question = question['question']
            tom_order = sum([1 if char in cur_question else 0 for char in character_lst])

            if 'bigtom' not in data_name:
                cur_instruction = instruction_template.replace('{{tom question}}', cur_question)
                cur_input_msg = deepcopy(few_shot_template)

                cur_input_msg.append({
                    'role': 'user',
                    'content': cur_instruction,
                })

                msg_lst.append(cur_input_msg)
                data[key]['questions'][idx]['reduced_question'] = ''

                msg_idx_to_question[msg_idx] = cur_question
                msg_idx += 1

            else:
                msg_idx_to_question[msg_idx] = cur_question
                cur_question = 'Which of the following statements is true?'
                data[key]['questions'][idx]['reduced_question'] = cur_question

    # FANToM questions do not need to be reduced
    if 'fantom' in data_name:
        for key, val in data.items():
            question_lst = val['questions']
            for idx, question in enumerate(tqdm(question_lst, position=1, leave=False, desc=f'Processing {key}')):
                    data[key]['questions'][idx]['reduced_question'] = question['question']
        return data

    result_lst = [model.inference(
        message=msg,
        model=model_name,
        temperature=0.0,
    ) for msg in msg_lst]
    result_lst = await tqdm.gather(
        *result_lst, 
        desc='Reducing TOM questions', 
        position=0, 
        leave=False
    )

    msg_idx = 0
    for key, val in data.items():
        question_lst = val['questions']
        for idx, question in enumerate(tqdm(question_lst, position=1, leave=False, desc=f'Processing {key}')):

            assert question['question'].lower() == msg_idx_to_question[msg_idx].lower()

            cur_result = result_lst[msg_idx]
            parsed_result = cur_result.split(':')[-1].strip()

            if not data[key]['questions'][idx]['reduced_question']:
                data[key]['questions'][idx]['reduced_question'] = parsed_result
            
            msg_idx += 1

    return data
