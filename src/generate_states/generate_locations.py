import os
import numpy as np

from components.utils import FileIO
from components.get_locations import LocationDetector
from components.llms import OpenAIInference, OpenAIAsyncInference


async def identify_locations(
    data: dict, 
    augmented_data: dict,
    data_name: str, 
    model: OpenAIInference | None, 
    async_model: OpenAIAsyncInference | None,
    model_name: str,
    prompt_path: str, 
    generation_config: dict, 
    seed: int,
    event_based:bool,
    to_jsonl: bool,
) -> None:

    file_io = FileIO()

    narrative_location_tracker = LocationDetector(
        model=model,
        async_model=async_model,
        model_name=model_name,
        prompt_path=prompt_path,
        dataset_name=data_name,
        generation_config=generation_config,
    )

    narrative_lst = []
    id_lst = []
    location_data = {id: {} for id in data.keys()}
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

    result_original, result_parsed, msg_lst, jsonl_out = await narrative_location_tracker.detect(
        narrative_lst=narrative_lst, 
        data_name=data_name,
        to_jsonl=to_jsonl,
        id_lst=id_lst,
    )

    if result_original:
        for j in range(len(id_lst)):
            location_data[id_lst[j]] = {
                'prompt': msg_lst[j][-1]['content'],
                'original_generation': result_original[j],
                'detected_locations': result_parsed[j],
            }

        postfix = ''
        postfix += '_event' if event_based else ''

        if not os.path.exists(f'../data/masktom/locations/{model_name}/'):
            os.makedirs(f'../data/masktom/locations/{model_name}/')

        file_io.save_json(
            location_data, 
            (
                f'../data/masktom/locations/{model_name}/' 
                f'{data_name}{postfix}_samples_seed[{seed}].json'
            )
        )
    else:
        file_io.save_jsonl(
            jsonl_out, 
            (
                f'../data/openai_batch_data/{model_name}/' 
                f'{data_name}_identify_location_seed{seed}.jsonl'
            )
        )
