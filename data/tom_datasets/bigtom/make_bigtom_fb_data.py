import json
import numpy as np


all_data = json.load(open('./bigtom_all.json'))


new_data = {}
used_ids = []
narrative2id = {}
counter = 0
for entry in all_data:
    if 'forward_belief' in entry['q_type']:

        counter += 1

        id = str(np.random.randint(10**9, 10**10))
        while id in used_ids:
            id = str(np.random.randint(10**9, 10**10))
        used_ids.append(id)

        cur_narrative = entry['narrative']

        cur_question = {
            'question': entry['question'],
            'true_answer': entry['true_answer'],
            'wrong_answer': entry['wrong_answer'],
            'q_type': entry['q_type'],
        }

        if cur_narrative not in narrative2id.keys():
            narrative2id[cur_narrative] = id

            new_data[id] = {
                'narrative': cur_narrative,
                'questions': [cur_question],
            }
        else:
            id = narrative2id[cur_narrative]

            new_data[id]['questions'].append(cur_question)

with open('./bigtom.json', 'w') as f:
    json.dump(new_data, f, indent=4)
f.close()

        