import os
from components.utils import FileIO
from components.llms import OpenAIInference, OpenAIAsyncInference
from components.get_entity_of_interest import EntityOfInterestExtractor

# setting logging level for HuggingFace Transformers
from transformers.utils import logging
logging.set_verbosity_error()


async def generate_entity_of_interest(
    data: dict, 
    augmented_data: dict,
    data_name: str, 
    model: OpenAIInference,
    async_model: OpenAIAsyncInference,
    model_name: str,
    prompt_path: str, 
    add_attr: bool,
    seed: int,
    event_based: bool,
    to_jsonl: bool,
):

    file_io = FileIO()
    character_info_lst = file_io.load_json(f'../data/masktom/ner/{data_name}_char_name_seed[{seed}].json')
    character_info_lst = list(character_info_lst.values())

    eoi_extractor = EntityOfInterestExtractor(
        model=model,
        async_model=async_model,
        model_name=model_name,
        prompt_path=prompt_path,
        dataset_name=data_name,
        add_attr=add_attr,
    )

    counter = 1
    narrative_lst = []
    question_lst = []
    id_lst = []
    for (id, entry) in data.items():

        if event_based:
            narrative = augmented_data[id]['events']
        elif data_name == 'fantom':
            narrative = entry['narrative'].split('\n')
            narrative = [ele for ele in narrative if ele.strip()]
        else:
            narrative = entry['narrative'] + ' '
            narrative = narrative.replace('\n\n', ' ')
            narrative = narrative.replace('\n', ' ')
            narrative = narrative.split('. ')
            narrative = [ele for ele in narrative if ele.strip()]
            narrative = [f'{idx+1}: {ele}.' for idx, ele in enumerate(narrative)]

        question = entry['questions']
        question = '\n'.join([f'"{ele["question"]}"' for ele in question])

        narrative_lst.append(narrative)
        question_lst.append(question)
        id_lst.append(id)

    result_original, result_parsed, mask_msg_list, jsonl_out = await eoi_extractor.extract(
        narrative=narrative_lst, 
        question_list=question_lst,
        char_lst=character_info_lst,
        to_jsonl=to_jsonl,
        id_lst=id_lst,
    )

    if to_jsonl:
        postfix = ''
        postfix += '_attr' if add_attr else ''
        file_io.save_jsonl(
            jsonl_out, 
            (
                f'../data/openai_batch_data/{model_name}/' 
                f'{data_name}_eoi{postfix}_seed{seed}.jsonl'
            )
        )
        file_io.save_json(
            character_info_lst,
            (
                f'../data/openai_batch_data/{model_name}/'
                f'{data_name}_eoi{postfix}_char_lst_seed{seed}.json'
            )
        )
        file_io.save_json(
            id_lst,
            (
                f'../data/openai_batch_data/{model_name}/'
                f'{data_name}_eoi{postfix}_id_lst_seed{seed}.json'
            )
        )
        file_io.save_json(
            question_lst,
            (
                f'../data/openai_batch_data/{model_name}/'
                f'{data_name}_eoi{postfix}_question_lst_seed{seed}.json'
            )
        )

    else:
        eoi_data = []
        j = 0
        for k, questions in enumerate(question_lst):
            gen = result_original[j]

            res = result_parsed[j]
            msg = mask_msg_list[j]

            eoi_data.append({
                'id': id_lst[k],
                'questions': questions,
                'prompt': msg[-1]['content'],
                'original_generation': gen,
                'response': res,
            })

            j += 1

        postfix = ''
        postfix += '_event' if event_based else ''
        postfix += '_attr' if add_attr else ''

        if not os.path.exists(f'../data/masktom/eoi/{model_name}'):
            os.makedirs(f'../data/masktom/eoi/{model_name}')

        file_io.save_json(
            eoi_data, 
            f'../data/masktom/eoi/{model_name}/{data_name}{postfix}_eoi_seed[{seed}].json'
        )
