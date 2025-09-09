import os, sys 
import numpy as np
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
    data = file_io.load_tsv(fpath)

    # NOTE: remove relations about "Physical Entity".
    # These properties are covered by OpenPI or are too fine-grained for current ToM benchmarks
    exclude_rels = ["ObjectUse", "isFilledBy", "AtLocation", "MadeUpOf", "HasProperty", "CapableOf", "Desires", "NotDesires"]
    
    print("\n[green]========= Data Statistics ===========[/green]")
    print(f"[green]Partition: {partition}[/green]")
    print(f"[green]Original Entries: {len(data)}[/green]")
    data = [row for row in data if row[1] not in exclude_rels]
    print(f"[green]Remaining Entries: {len(data)}[/green]")
    print("[green]=====================================[/green]\n")

    data_lst = []
    counter = 0
    prev_head = None
    for entry in data:
        if entry[0].strip()[-1] != '.': 
            entry[0] = entry[0].strip() + '.'

        if '___' in entry[0] and entry[0] != prev_head:
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

    prev_head = None
    augment_entries = {}
    counter = 0
    for key, val in new_data.items():
        if '___' in key:
            cur_head, cur_rel = key.split('. ')
            cur_rel = cur_rel.split(' [GEN]')[0]

            if cur_head not in augment_entries.keys():
                augment_entries[cur_head] = []
                augment_entries[cur_head].append({'rel': cur_rel, 'val': val})
            else:
                augment_entries[cur_head].append({'rel': cur_rel, 'val': val})

    for key, val in augment_entries.items():

        print(key)
        print(val)

        raise SystemExit()

    print(f"[red]Augment {len(augment_entries)} entries[/red]")
    print("[green]\nLoading Mixtral model for augmentation...[/green]")

    global mixtral_model
    BATCH_SIZE = 20
    mixtral_model = HFInference()
    mixtral_model.init_model('mixtral', config_path='./mixtral_config.yaml', batch_size=BATCH_SIZE)

    # NOTE: introduce human-readable relations for augmenting entries
    atomic_human_readable_rel = HumanReadable()

    # NOTE: set current batch size to be 5
    head_batch, rel_batch, tail_batch = [], [], []
    aug_head_lst = []
    for idx, (cur_head, val) in enumerate(seacow_progress(augment_entries.items(), "Augmenting Entries...", "red")):

        sampled_rel = np.random.choice(val)
        while 'none' in sampled_rel['val']:
            sampled_rel = np.random.choice(val)

        cur_rel = sampled_rel['rel']  
        cur_tail = sampled_rel['val']
        cur_rel = atomic_human_readable_rel.get_human_readable(cur_rel)

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

    file_io.save_json(aug_head_lst, f'./atomic_head_augment_{partition}.json')


def main():

    format_data('./train.tsv')
    format_data('./dev.tsv')
    format_data('./test.tsv')


if __name__ == '__main__':
    main()

