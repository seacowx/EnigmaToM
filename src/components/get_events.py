import os
import asyncio
from copy import deepcopy
from pydantic import BaseModel
from json_repair import repair_json

from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async

from components.llms import OpenAIInference
from components.utils import FileIO, load_model


def compute_lexical_overlap(prev_sent: str, cur_sent: str) -> float:
    prev_words = prev_sent.split()
    cur_words = cur_sent.split()
    return len(set(prev_words) & set(cur_words)) / len(set(prev_words) | set(cur_words))


class ProfileSchema(BaseModel):
    Name: str
    Occupation: list
    Preference: list


async def extract_events(
    data: dict,
    character_dict: dict,
    data_name: str,
    model_name: str,
    prompt_dict: dict,
    seed: int,
    no_async: bool,
) -> dict:

    model = load_model(
        model_name=model_name,
        no_async=no_async,
    )

    file_io = FileIO()
    event_template = prompt_dict['extract_events']

    if 'gemma' in model_name:
        event_msg_template = []
    else:
        event_msg_template = [{'role': 'system', 'content': prompt_dict['extract_events_system']}]

    profile_template = prompt_dict['character_profile_extraction']
    profile_msg_template = [{'role': 'system', 'content': prompt_dict['character_profile_extraction_system']}]
    backoff_profile = prompt_dict['backoff_profile']

    event_msg_lst = []
    # event_msg_idx_to_id = {}
    # msg_idx = 0
    for key, val in data.items():
        narrative = val['narrative']
        cur_char_lst = character_dict[key]

        if data_name == 'fantom':
            narrative_sents = narrative.split('\n')
        else:
            narrative += ' '
            narrative = narrative.replace('\n\n', ' ')
            narrative = narrative.replace('\n', ' ')
            narrative_sents = narrative.split('. ')
            narrative_sents = [ele + '.' for ele in narrative_sents if ele.strip()]
            narrative_sents = [ele.replace('..', '.') for ele in narrative_sents]
            narrative = '\n'.join(
                [f'{idx+1}: {ele}' for idx, ele in enumerate(narrative_sents)]
            )

        # for sent in narrative_sents:

        #     cur_template = event_template.replace('{{sentence}}', sent)
        #     cur_msg = event_msg_template + [{'role': 'user', 'content': cur_template}]
        #     event_msg_lst.append(cur_msg)
        #     event_msg_idx_to_id[msg_idx] = key
        #     msg_idx += 1

        cur_template = event_template.replace('{{narrative}}', narrative)
        cur_msg = event_msg_template + [{'role': 'user', 'content': cur_template}]
        event_msg_lst.append(cur_msg)

    # generate responses via async inference
    if no_async:
        cur_event_lst = []
        for cur_msg in tqdm(event_msg_lst):
            cur_event = model.inference(
                model=model_name,
                message=cur_msg,
                temperature=0.,
            )
            cur_event_lst.append(cur_event)
    else:
        semaphore = asyncio.Semaphore(10)
        cur_event_lst = [model.process_with_semaphore(
            semaphore=semaphore,
            model=model_name,
            message=cur_msg[:10],
            temperature=0.,
        ) for cur_msg in event_msg_lst]
        cur_event_lst = await tqdm_async.gather(*cur_event_lst)

    # # reassamble events into narrative
    # event_idx_dict = {}
    # augmented_data = {}
    # for cur_msg_idx, response in enumerate(cur_event_lst):

    #     cur_id = event_msg_idx_to_id[cur_msg_idx]
    #     if cur_id not in event_idx_dict:
    #         event_idx_dict[cur_id] = 1

    #     if cur_id not in augmented_data:
    #         augmented_data[cur_id] = {
    #             'narrative': data[cur_id]['narrative'],
    #             'events': {},
    #         }

    #     event = response.rsplit('Key Event:', 1)[-1]
    #     if event.startswith('-'):
    #         event = event.split('-', 1)[1].strip()
    #     if event[0].isdigit() or ':' in event[:20] or '.' in event[:20]:
    #         event = event.split('.')[-1].split(':')[-1].strip()

    #     cur_event_idx = event_idx_dict[cur_id]
    #     augmented_data[cur_id]['events'][cur_event_idx] = event.strip()
    #     event_idx_dict[cur_id] += 1

    augmented_data = {}
    need_fix = {}
    for key, response, usr_msg in zip(data, cur_event_lst, event_msg_lst):
        if '<Key Events>' in response:
            event_lst = response.rsplit('<Key Events>', 1)[1].strip().split('\n')
        else:
            event_lst = response.split('\n')
            event_lst = [ele for ele in event_lst if ele.strip()]
            event_lst = [ele for ele in event_lst if ele.strip()[0].isdigit()]

        if not event_lst:
            need_fix[key] = usr_msg

        event_lst = [ele for ele in event_lst if ele.strip()]
        event_lst = [ele for ele in event_lst if ele[0].isdigit()]
        augmented_data[key] = {
            'narrative': data[key]['narrative'],
            'events': event_lst,
        }

        fixed_keys = []
        while len(need_fix):

            print(f'Fixing {len(need_fix)} entries.')
            fix_msg_lst = list(need_fix.values())

            cur_event_lst = [model.inference(
                model=model_name,
                message=cur_msg,
                temperature=0.5,
            ) for cur_msg in fix_msg_lst]
            cur_event_lst = await tqdm_async.gather(*cur_event_lst)

            for key, response, usr_msg in zip(need_fix, cur_event_lst, fix_msg_lst):
                if '<Key Events>' in response:
                    event_lst = response.rsplit('<Key Events>', 1)[1].strip().split('\n')
                else:
                    event_lst = response.split('\n')
                    event_lst = [ele for ele in event_lst if ele.strip()]
                    event_lst = [ele for ele in event_lst if ele.strip()[0].isdigit()]

                if not event_lst:
                    need_fix[key] = usr_msg
                else:
                    fixed_keys.append(key)

                event_lst = [ele for ele in event_lst if ele.strip()]
                event_lst = [ele for ele in event_lst if ele[0].isdigit()]
                augmented_data[key] = {
                    'narrative': data[key]['narrative'],
                    'events': event_lst,
                }

            # exclude fixed entries
            need_fix = {k: v for k, v in need_fix.items() if k not in fixed_keys}

    if not os.path.exists(f'../data/augmented_data/{model_name}'):
        os.makedirs(f'../data/augmented_data/{model_name}')

    file_io.save_json(
        augmented_data, 
        f'../data/augmented_data/{model_name}/{data_name}_augmented_seed[{seed}].json'
    )
    return augmented_data
