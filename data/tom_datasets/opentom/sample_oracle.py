import json
from copy import deepcopy


with open('./opentom_reduced_q_seed[12].json') as f:
    data = json.load(f)

with open('../../augmented_data/opentom_augmented_seed[12].json', 'r') as f:
    aug_data = json.load(f)


out_data = {}
for key, val in data.items():

    idx = 0
    mover = val['plot_info']['mover']
    affected_char = val['plot_info']['affected_char']
    cur_characters = [mover, affected_char]

    aug_narrative = '\n'.join(aug_data[key]['events'])

    out_data[key] = {
        mover: aug_narrative,
        affected_char: aug_narrative
    }

    # out_data[f'{key}_{mover}'] = deepcopy(val)
    # out_data[f'{key}_{mover}']['questions'] = []
    # out_data[f'{key}_{affected_char}'] = deepcopy(val)
    # out_data[f'{key}_{affected_char}']['questions'] = []
    # out_data[f'{key}_narrator'] = deepcopy(val)
    # out_data[f'{key}_narrator']['questions'] = []

    # cur_question_lst = val['questions']
    # for q_dict in cur_question_lst:

    #     if q_dict['type'] == 'attitude':
    #         q_dict['oracle_narrative'] = aug_narrative
    #         out_data[f'{key}_{affected_char}']['questions'].append(q_dict)

    #     elif q_dict['type'] == 'location-fo':
    #         q_dict['oracle_narrative'] = aug_narrative
    #         cur_char = q_dict['question'].split("'s", 1)[0].replace('From ', '').strip()
    #         out_data[f'{key}_{cur_char}']['questions'].append(q_dict)

    #     elif q_dict['type'] == 'location-so':
    #         q_dict['oracle_narrative'] = aug_narrative
    #         cur_char = q_dict['question'].split("'s", 1)[0].replace('From ', '').strip()
    #         out_data[f'{key}_{cur_char}']['questions'].append(q_dict)

    #     elif q_dict['type'] == 'multihop-fo':
    #         q_dict['oracle_narrative'] = aug_narrative
    #         cur_char = q_dict['question'].split("'s", 1)[0].replace('From ', '').strip()
    #         out_data[f'{key}_{cur_char}']['questions'].append(q_dict)
        
    #     elif q_dict['type'] == 'multihop_so':
    #         q_dict['oracle_narrative'] = aug_narrative
    #         cur_char = q_dict['question'].split("'s", 1)[0].replace('From ', '').strip()
    #         out_data[f'{key}_{cur_char}']['questions'].append(q_dict)

with open('./opentom_oracle_sample_12.json', 'w') as f:
    json.dump(out_data, f, indent=4)
