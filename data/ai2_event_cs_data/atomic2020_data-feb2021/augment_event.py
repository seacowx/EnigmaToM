import re
import os, sys 
import numpy as np
from random import shuffle
from unidecode import unidecode
sys.path.append('../../../src/')

from glob import glob
from rich import print
from rich.markup import escape

from typing import List, Dict

from utils import FileIO, seacow_progress


def format_data(data):
    # NOTE: remove relations about "Physical Entity".
    # These properties are covered by OpenPI or are too fine-grained for current ToM benchmarks
    exclude_rels = ["ObjectUse", "isFilledBy", "AtLocation", "MadeUpOf", "HasProperty", "CapableOf", "Desires", "NotDesires"]
    
    print("\n[green]========= Data Statistics ===========[/green]")
    print(f"[green]Original Entries: {len(data)}[/green]")
    data = [row for row in data if row[1] not in exclude_rels]
    print(f"[green]Remaining Entries: {len(data)}[/green]")
    print("[green]=====================================[/green]\n")

    data_lst = []
    counter = 0
    prev_head = None
    for entry in seacow_progress(data, "Formatting ATOMIC data", "green"):
        if entry[0].strip()[-1] != '.': 
            entry[0] = entry[0].strip() + '.'

        #if '___' in entry[0] and entry[0] != prev_head:
        if entry[0] != prev_head:
            prev_head = entry[0]
            counter += 1

        data_lst.append({
            'instruction': entry[0] + ' ' + entry[1].strip() + ' [GEN]',
            'response': entry[2].strip()
        })

    new_data = {}
    for entry in data_lst:

        cur_key = entry['instruction'].strip()

        if cur_key not in new_data.keys():
            new_data[entry['instruction']] = []

        new_data[cur_key].append(entry['response'])

    for key, val in new_data.items():
        new_data[key] = list(set(val))

        if len(val) > 1 and 'none' in val:
            new_data[key].remove('none')

        if not new_data[key]:
            new_data[key] = ['none']

    new_atomic_data = []
    for key, val in new_data.items():
        cur_head, cur_rel = key.split('. ')
        cur_rel = cur_rel.split(' [GEN]')[0]

        new_atomic_data.append({
            'head': cur_head,
            'rel': cur_rel,
            'tail': val,
        })

    return new_atomic_data


def parse_response(response_dict: Dict[str, str]) -> List[str]:
    '''
    parse_mixtral_response parse the augmentation response from Mixtral

    Args:
        response (str): the original Mixtral response

    Returns:
        List[str]: list of augmented words
    '''

    response = unidecode(response_dict['augmented'])

    if response == '[\'none\']':
        return ['none']

    main_pattern = r"\*?\s?\[?\s?([\"\w\s\'\/\-\"]+),\s?([\"\w\s\'\/\-\"]+),\s?([\"\w\s\'\/\-\"]+),\s?([\"\w\s\'\/\-\"]+),\s?([\"\w\s\'\/\-\"]+)\]?"
    alternative_pattern = r'"(.*?)"'
    main_pattern = re.compile(main_pattern)
    alternative_pattern = re.compile(alternative_pattern)

    response_match = main_pattern.match(response.strip())

    if response_match:
        response_list = list(response_match.groups())
        if '[' in response_list[0]:
            response_list[0] = response_list[0].split('[')[1]
        if ']' in response_list[-1]:
            response_list[-1] = response_list[-1].split(']')[0]
        
        response_list = [ele.strip().lower() for ele in response_list]

    else:
        if '\n' in response:
            response_list = response.split('\n')
            response_list = [ele.split('.')[-1].strip().lower() for ele in response_list]

        else:
            response_match = alternative_pattern.findall(response.strip())

            if response_match:
                response_list = [ele.strip().lower() for ele in response_match]
            else:
                return ['none']

    response_list = [ele.replace("'", '').replace('"', '') for ele in response_list if ele != '']
    return response_list


def augment_data(fpath: str, head_augment_fpath: str):

    atomic_data = file_io.load_tsv(fpath)
    atomic_data = format_data(atomic_data)

    atomic_head_aug_data = file_io.load_json(head_augment_fpath)

    atomic_head_aug_keys = [ele['original'] for ele in atomic_head_aug_data]

    # TODO: replace PersonX PersonY, etc with random names
    # TODO: fill in blank with atomic_head_augment_data
    corrupted_idx = []
    num_modified = 0
    for idx, entry in enumerate(seacow_progress(atomic_data, 'Augmenting ATOMIC data', 'green')):
        cur_head, cur_rel, cur_tail = entry.values()

        atomic_data[idx]['head'] = cur_head

        # NOTE: add relationship to atomic data dict
        atomic_data[idx]['rel'] = cur_rel

        # NOTE: fill in the blank (___) in the ATOMIC head events
        if cur_head in atomic_head_aug_keys:

            cur_head_aug_entry = atomic_head_aug_data[atomic_head_aug_keys.index(cur_head)]
            augment_words = parse_response(cur_head_aug_entry)

            if augment_words == ['none']:
                corrupted_idx.append(idx)
                continue
            else:
                original_head = atomic_data[idx]['head']
                aug_word = np.random.choice(augment_words)
                filled_head = original_head.replace('___', aug_word.strip())

                atomic_data[idx]['head'] = filled_head

                num_modified += 1

    print(f"[green]======================================================[/green]")
    print(f"[green]Original ATOMIC data size: {len(atomic_data)}[/green]")
    print(f"[red]{len(corrupted_idx)} entries are corrupted[/red]")
    print(f"[green]======================================================[/green]")

    # NOTE: remove entries where the augmentation result is corrupted
    atomic_data = [ele for idx, ele in enumerate(atomic_data) if idx not in corrupted_idx]

    print(f"[green]======================================================[/green]")
    print(f"[green]ATOMIC data size after augmentation: {len(atomic_data)}[/green]")
    print(f"[green]======================================================[/green]")

    data_split = fpath.split('/')[-1].split('.')[0]
    save_path = f'./atomic_{data_split}.json'
    print(f"[green]File saved at: {save_path}[/green]")
    file_io.save_json(atomic_data, save_path) 


def main():

    global file_io
    file_io = FileIO()

    augment_data(
        './train.tsv', 
        './atomic_head_augment_train.json',
    )

    augment_data(
        './test.tsv', 
        './atomic_head_augment_test.json',
    )

    augment_data(
        './dev.tsv', 
        './atomic_head_augment_dev.json',
    )

if __name__ == '__main__':
    main()
