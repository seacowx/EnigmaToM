import re
import ast
import os, sys 
import argparse
import numpy as np
from copy import deepcopy
from random import shuffle
from unidecode import unidecode
sys.path.append('../../../src/')

from glob import glob
from rich import print
from rich.markup import escape

from typing import List, Dict

from utils import FileIO, seacow_progress


def format_data(data) -> tuple:
    # NOTE: remove relations about "Physical Entity".
    # These properties are covered by OpenPI or are too fine-grained for current ToM benchmarks
    
    print("\n[green]========= Data Statistics ===========[/green]")
    print(f"[green]Number of Data Entries: {len(data)}[/green]")
    print("[green]=====================================[/green]\n")

        # NOTE: header conntent:
    rels = ['Xintent', 'Xemotion', 'Oemotion']
    new_data = {}
    visited_head_lst = []
    for entry in seacow_progress(data[1:], 'Formatting data', 'green'):

        head = entry[1]

        if head.strip()[-1] != '.':
            head = head.strip() + '.'

        if head not in visited_head_lst:
            visited_head_lst.append(head)
            new_data[head] = {r: ['none'] for r in rels}

        for rel, tail in zip(rels, entry[2:5]):

            if 'none' not in tail:
                if 'none' in new_data[head][rel]:
                    new_data[head][rel].remove('none')
                new_data[head][rel].extend(ast.literal_eval(tail))

    # NOTE: remove entries with no response
    print("\n[green]========= Data Statistics ===========[/green]")
    print(f"[green]Validated Entries: {len(new_data)}[/green]")
    print("[green]=====================================[/green]\n")

    event2mind_data, event2mind_data_grouped = [], []
    for key, val in new_data.items():
        if key.strip()[-1] != '.':
            key = key.strip() + '.'
        cur_head = key

        all_rel_tail_pairs = []
        for rel, tail in val.items():
            event2mind_data.append({
                'head': cur_head,
                'rel': rel,
                'tail': str(tail),
            })

            all_rel_tail_pairs.append(
                f"{rel}\n{str(tail)}"
            )

        event2mind_data_grouped.append({
            'head': cur_head,
            'rel': '\n\n'.join(all_rel_tail_pairs),
        })

    return event2mind_data, event2mind_data_grouped


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

    event2mind_data = file_io.load_csv(fpath)
    event2mind_data, event2mind_data_grouped = format_data(event2mind_data)

    event2mind_head_aug_data = file_io.load_json(head_augment_fpath)

    event2mind_head_aug_keys = [ele['original'] for ele in event2mind_head_aug_data]

    if args.grouped:
        event2mind_data = event2mind_data_grouped

    corrupted_idx = []
    num_modified = 0
    for idx, entry in enumerate(seacow_progress(event2mind_data, 'Augmenting Event2Mind data', 'green')):

        if args.grouped:
            cur_head, cur_rel = entry.values()
        else:
            cur_head, cur_rel, _ = entry.values()

        event2mind_data[idx]['head'] = cur_head

        # NOTE: add relationship to Event2Mind data dict
        event2mind_data[idx]['rel'] = cur_rel

        # NOTE: fill in the blank (___) in the Event2Mind head events
        if cur_head in event2mind_head_aug_keys:

            cur_head_aug_entry = event2mind_head_aug_data[event2mind_head_aug_keys.index(cur_head)]
            augment_words = parse_response(cur_head_aug_entry)

            if augment_words == ['none']:
                corrupted_idx.append(idx)
                continue
            else:
                original_head = event2mind_data[idx]['head']
                aug_word = np.random.choice(augment_words)
                filled_head = original_head.replace('___', aug_word.strip())

                event2mind_data[idx]['head'] = filled_head

                num_modified += 1

    print(f"[green]======================================================[/green]")
    print(f"[green]Original Event2Mind data size: {len(event2mind_data)}[/green]")
    print(f"[red]{len(corrupted_idx)} entries are corrupted[/red]")
    print(f"[green]======================================================[/green]")

    # NOTE: remove entries where the augmentation result is corrupted
    event2mind_data = [ele for idx, ele in enumerate(event2mind_data) if idx not in corrupted_idx]

    print(f"[green]======================================================[/green]")
    print(f"[green]Event2Mind data size after augmentation: {len(event2mind_data)}[/green]")
    print(f"[green]======================================================[/green]")

    data_split = fpath.split('/')[-1].split('.')[0]

    if args.grouped:
        save_path = f'./event2mind_{data_split}_grouped.json'
    else:
        save_path = f'./event2mind_{data_split}.json'

    print(f"[green]File saved at: {save_path}[/green]")
    file_io.save_json(event2mind_data, save_path) 


def get_args():
    parser = argparse.ArgumentParser(description='Augment Event2Mind data')
    parser.add_argument(
        '--grouped', action='store_true', help='whether to group data by head'
    )
    return parser.parse_args()


def main():

    global args, file_io
    args = get_args()
    file_io = FileIO()

    augment_data(
        './train.csv', 
        './event2mind_head_augment_train.json',
    )

    augment_data(
        './test.csv', 
        './event2mind_head_augment_test.json',
    )

    augment_data(
        './dev.csv', 
        './event2mind_head_augment_dev.json',
    )

if __name__ == '__main__':
    main()
