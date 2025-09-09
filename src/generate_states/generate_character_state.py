import os
import numpy as np

from components.utils import FileIO, seacow_progress
from components.get_character_loc import CharacterLocationTracker
from components.llms import OpenAIInference, OpenAIAsyncInference


async def generate_character_state(
    data: dict, 
    augmented_data: dict,
    data_name: str, 
    model: OpenAIInference,
    async_model: OpenAIAsyncInference,
    model_name: str,
    prompt_path: str, 
    generation_config: dict, 
    seed: int,
    event_based: bool,
    to_jsonl: bool,
) -> None:

    file_io = FileIO()

    data_character_lst = file_io.load_json(
        f'../data/masktom/ner/{data_name}_char_name_seed[{seed}].json'
    )

    character_location_tracker = CharacterLocationTracker(
        model=model,
        async_model=async_model,
        model_name=model_name,
        prompt_path=prompt_path,
        dataset_name=data_name,
        generation_config=generation_config, 
        seed=seed,
    )

    narrative_lst = []
    narrative_len_lst = []
    character_lst = []
    id_lst = []
    for ((id, entry), char_lst) in zip(data.items(), data_character_lst.values()):

        if event_based:
            indexed_n = augmented_data[id]['events']
        elif data_name == 'fantom':
            indexed_n = entry['narrative'].split('\n')
            indexed_n = [ele for ele in indexed_n if ele.strip()]
            indexed_n = [f'{idx+1}: {ele}.' for idx, ele in enumerate(indexed_n)]
        else:
            indexed_n = entry['narrative'] + ' '
            indexed_n = indexed_n.replace('\n\n', ' ')
            indexed_n = indexed_n.replace('\n', ' ')
            indexed_n = indexed_n.split('. ')
            indexed_n = [ele for ele in indexed_n if ele.strip()]
            indexed_n = [f'{idx+1}: {ele}.' for idx, ele in enumerate(indexed_n)]

        narrative_len_lst.append(len(indexed_n))
        narrative_lst.append('\n'.join(indexed_n))
        character_lst.append(char_lst)
        id_lst.append(id)

    char_loc_res = await character_location_tracker.track(
        narrative=narrative_lst, 
        character_list=character_lst,
        id_lst=id_lst,
        to_jsonl=to_jsonl,
    )
    result_original, result_parsed, result_vecs, mask_msg_list, narrative_len_lst, jsonl_out = char_loc_res

    if jsonl_out:
        file_io.save_jsonl(
            jsonl_out, 
            (
                f'../data/openai_batch_data/{model_name}/' 
                f'{data_name}_character_state_seed{seed}.jsonl'
            )
        )
        file_io.save_json(
            narrative_len_lst,
            (
                f'../data/openai_batch_data/{model_name}/' 
                f'{data_name}_narrative_len_lst_seed{seed}.json'
            )
        )
        file_io.save_json(
            character_lst,
            (
                f'../data/openai_batch_data/{model_name}/' 
                f'{data_name}_character_state_character_lst_seed{seed}.json'
            )
        )
    else:
        j = 0
        perception_data = []
        for k, chars in enumerate(character_lst):

            for char in chars:
                gen = result_original[j]
                res = result_parsed[j]
                msg = mask_msg_list[j]
                vec = result_vecs[j]

                perception_data.append({
                    'id': id_lst[k],
                    'character': char,
                    'prompt': msg[-1]['content'],
                    'original_generation': gen,
                    'response': res,
                    'location_vec': vec, 
                })

                j += 1

        postfix = ''
        postfix += '_event' if event_based else ''

        if not os.path.exists(f'../data/masktom/character_state/{model_name}'):
            os.makedirs(f'../data/masktom/character_state/{model_name}')

        file_io.save_json(
            perception_data, 
            f'../data/masktom/character_state/{model_name}/{data_name}{postfix}_samples_seed[{seed}].json'
        )
