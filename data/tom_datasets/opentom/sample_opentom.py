import json
import numpy as np


np.random.seed(2024)


data = json.load(open('./opentom_full.json'))

for key, val in data.items():

    cur_questions = val['questions']

    q_type_dict = {}

    for q_dict in cur_questions:
        if q_dict['type'] not in q_type_dict:
            q_type_dict[q_dict['type']] = [q_dict]
        else:
            q_type_dict[q_dict['type']] += [q_dict]

    new_questions = []
    for q_type, q_list in q_type_dict.items():
        if 'location' not in q_type:
            continue
        np.random.shuffle(q_list)
        new_questions.extend(q_list[:4])

    data[key]['questions'] = new_questions


json.dump(data, open('./opentom.json', 'w'), indent=4)