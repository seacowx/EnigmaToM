import os
import numpy as np
from copy import deepcopy

from components.llms import OpenAIInference, OpenAIAsyncInference
from components.utils import FileIO
from components.get_narrative_locations import NarrativeLocationTracker


def make_narrative_location_vec(
    data: list, 
    narrative_lst: list,
):

    for data_idx, entry in enumerate(data):

        narrative = narrative_lst[data_idx]

        narrative_len = len(narrative)

        response = entry['response']
        narrative_location_vec = ['none'] * narrative_len

        prev_idx = 0
        prev_location = 'none'
        last_visited_loc = ''

        for loc_dict in response:

            sent_idx, current_loc = loc_dict.values()
            sent_idx = int(sent_idx) - 1
            current_loc = current_loc.split('(', 1)[0].strip().lower()

            # the case where llm generates more locations than the length of the narrative
            if sent_idx >= narrative_len:
                for idx in range(prev_idx, narrative_len):
                    narrative_location_vec[idx] = prev_location

            # the case where llm generates locations within the length of the narrative
            else:
                for idx in range(prev_idx, sent_idx):
                    narrative_location_vec[idx] = prev_location
                prev_idx = sent_idx

            last_visited_loc = current_loc
            prev_location = current_loc

        # fill in the remaining locations
        for idx in range(prev_idx, narrative_len):
            narrative_location_vec[idx] = prev_location

        try:
            last_idx = int(response[-1]['line_number'])
        except:
            continue

        # the case where llm generates fewer locations than the length of the narrative
        if last_idx  < narrative_len:
            for idx in range(last_idx, narrative_len):
                narrative_location_vec[idx] = last_visited_loc

        data[data_idx]['narrative_location_vec'] = narrative_location_vec

    return data


async def generate_narrative_location(
    data: dict, 
    augmented_data: dict,
    data_name: str, 
    model: OpenAIInference,  
    async_model: OpenAIAsyncInference,
    model_name: str,
    prompt_path: str, 
    seed: int,
    event_based: bool,
    to_jsonl: bool,
) -> None:

    file_io = FileIO()

    narrative_location_tracker = NarrativeLocationTracker(
        model=model,
        async_model=async_model,
        model_name=model_name,
        prompt_path=prompt_path,
        dataset_name=data_name,
        seed=seed,
    )

    narrative_lst = []
    id_lst = []
    for (id, entry) in data.items():

        if event_based:
            narrative = augmented_data[id]['events']
        elif data_name == 'fantom':
            narrative = entry['narrative'].split('\n')
            narrative = [ele for ele in narrative if ele.strip()]
            narrative = [f'{idx+1}: {ele}.' for idx, ele in enumerate(narrative)]
        else:
            narrative = entry['narrative'] + ' '
            narrative = narrative.replace('\n\n', ' ')
            narrative = narrative.replace('\n', ' ')
            narrative = narrative.split('. ')
            narrative = [ele for ele in narrative if ele.strip()]
            narrative = [f'{idx+1}: {ele}.' for idx, ele in enumerate(narrative)]

        narrative_lst.append(narrative)
        id_lst.append(id)

    nar_loc_res = await narrative_location_tracker.track(
        id_lst=id_lst, 
        narrative=narrative_lst, 
        data_name=data_name,
        to_jsonl=to_jsonl,
    )
    
    (
        result_original, 
        result_parsed, 
        msg_list, 
        available_location_lst, 
        jsonl_out, 
        automatic_idx_lst, 
        narrative_loc_result_automatic
    ) = nar_loc_res

    if jsonl_out:
        file_io.save_jsonl(
            jsonl_out, 
            (
                f'../data/openai_batch_data/{model_name}/' 
                f'{data_name}_narrative_location_seed{seed}.jsonl'
            )
        )
        file_io.save_json(
            id_lst,
            (
                f'../data/openai_batch_data/{model_name}/' 
                f'{data_name}_narrative_location_id_lst_seed{seed}.json'
            )
        )
        file_io.save_json(
            automatic_idx_lst,
            (
                f'../data/openai_batch_data/{model_name}/' 
                f'{data_name}_narrative_location_automatic_idx_lst_seed{seed}.json'
            )
        )
        file_io.save_json(
            narrative_lst,
            (
                f'../data/openai_batch_data/{model_name}/' 
                f'{data_name}_narrative_location_narrative_lst_seed{seed}.json'
            )
        )
        file_io.save_json(
            narrative_loc_result_automatic,
            (
                f'../data/openai_batch_data/{model_name}/' 
                f'{data_name}_narrative_location_result_automatic_seed{seed}.json'
            )
        )
        file_io.save_json(
            available_location_lst,
            (
                f'../data/openai_batch_data/{model_name}/' 
                f'{data_name}_narrative_location_available_location_lst_seed{seed}.json'
            )
        )
    else:
        location_data = []
        for j in range(len(id_lst)):
            location_data.append({
                'id': id_lst[j],
                'prompt': msg_list[j][-1]['content'],
                'original_generation': result_original[j],
                'response': result_parsed[j],
                'available_location': available_location_lst[j],
            })

        location_data = make_narrative_location_vec(
            data=location_data,
            narrative_lst=narrative_lst,
        )

        postfix = ''
        postfix += '_event' if event_based else ''

        if not os.path.exists(f'../data/masktom/narrative_loc/{model_name}/'):
            os.makedirs(f'../data/masktom/narrative_loc/{model_name}/')

        file_io.save_json(
            location_data, 
            (
                f'../data/masktom/narrative_loc/{model_name}/'
                f'{data_name}{postfix}_samples_seed[{seed}].json'
        ))
