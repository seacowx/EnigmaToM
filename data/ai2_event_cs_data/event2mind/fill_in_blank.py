import ast
import os, sys 
import numpy as np
from copy import deepcopy
sys.path.append('../../src/')

import string
from rich import print
from rich.markup import escape

from typing import List

from utils import FileIO, seacow_progress
from components.llms import HFInference
from data_storage import HumanReadable, augmentation_prompt


def extract_first_span(text):
    start, end = None, None
    for i in range(len(text)): 
        if text[i] not in string.punctuation:
            start = i
            break

    for i in range(start, len(text)):
        if text[i] in string.punctuation:
            end = i
            break

    if end:
        return text[start:end] 
    else:
        return text[start:]


def parse_mixtral_response(response: str) -> str:
    '''
    Parse the response from Mixtral model
    '''
    response = response.split(':')[-1].strip()

    if response:
        return response
    else:
        return "['none']"


def augment_entry(head: List[str], rel: List[str], tail: List[List[str]]) -> List[str]:
    '''
    Augment the entry with human-readable relations
    '''
    prompt = augmentation_prompt(head, rel, tail)

    response = mixtral_model.pipeline_inference(prompt)

    response = [parse_mixtral_response(resp) for resp in response]

    return response


def format_data(fpath: str):
    file_io = FileIO()

    partition = fpath.split('/')[-1].split('.')[0].strip()
    data = file_io.load_csv(fpath)
    
    print("\n[green]========= Data Statistics ===========[/green]")
    print(f"[green]Original Entries: {len(data)}[/green]")
    print("[green]=====================================[/green]\n")


    # NOTE: header conntent:
    rels = ['Xintent', 'Xemotion', 'Oemotion']
    new_data = {}
    visited_head_lst = []
    for entry in data[1:]:

        head = entry[1]

        if head.strip()[-1] != '.':
            head = head.strip() + '.'

        if head not in visited_head_lst:
            visited_head_lst.append(head)
            new_data[head] = {r: [] for r in rels}

        for rel, tail in zip(rels, entry[2:5]):
            if 'none' not in tail:
                new_data[head][rel].append(ast.literal_eval(tail))

    for key, val in new_data.items():

        temp_val = deepcopy(val)
        for rel, tail in val.items():

            if not tail:
                del temp_val[rel]
            else:
                temp_val[rel] = list(set(sum(tail, [])))

        new_data[key] = temp_val

    # NOTE: remove entries with no response
    print("\n[green]========= Data Statistics ===========[/green]")
    print(f"[green]Validated Entries: {len(new_data)}[/green]")
    print("[green]=====================================[/green]\n")

    augment_entries = {}
    for key, val in new_data.items():

        if '___' in key:
            augment_entries[key] = val


    print(f"[red]Augment {len(augment_entries)} entries[/red]")
    print("[green]\nLoading Mixtral model for augmentation...[/green]")

    global mixtral_model
    BATCH_SIZE = 20
    mixtral_model = HFInference()
    mixtral_model.init_model('mixtral', config_path='./mixtral_config.yaml', batch_size=BATCH_SIZE)

    # NOTE: introduce human-readable relations for augmenting entries
    event2mind_human_readable = HumanReadable()

    # NOTE: set current batch size to be 5
    head_batch, rel_batch, tail_batch = [], [], []
    aug_head_lst = []
    for idx, (cur_head, val) in enumerate(seacow_progress(augment_entries.items(), "Augmenting Entries...", "red")):

        sampled_rel = np.random.choice(list(val.keys()))
        sampled_rel = {'rel':sampled_rel, 'val': val[sampled_rel]}

        cur_rel = sampled_rel['rel']  
        cur_tail = sampled_rel['val']
        cur_rel = event2mind_human_readable.get_human_readable(cur_rel)

        # NOTE: the relationship must be included in the ATOMIC dataset
        assert cur_rel != '[UNK]', f"Unknown relation: {cur_rel}"

        if idx % BATCH_SIZE == 0 and idx:
            new_head = augment_entry(head_batch, rel_batch, tail_batch)
            
            for j, head in enumerate(head_batch):
                aug_head_lst.append({
                    'original': head,
                    'augmented': new_head[j],
                    'relation': rel_batch[j],
                    'tail': tail_batch[j]
                })

            head_batch, rel_batch, tail_batch = [cur_head], [cur_rel], [cur_tail]

        else:
            head_batch.append(cur_head)
            rel_batch.append(cur_rel)
            tail_batch.append(cur_tail)

    file_io.save_json(aug_head_lst, f'./event2mind_head_augment_{partition}.json')


def main():

    # format_data('./train.csv')
    # format_data('./dev.csv')
    format_data('./test.csv')


if __name__ == '__main__':
    main()

