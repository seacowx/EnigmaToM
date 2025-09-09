import json
import numpy as np


data = json.load(open('./tomi/tomi.json'))
data = json.load(open('./tomi/tomi_original.json'))

unique_narrative = []
new_data = {}
narrative2id = {}
used_ids = []
for entry in data:
    if entry['narrative'] not in unique_narrative:

        id = str(np.random.randint(10**9, 10**10))
        while id in used_ids:
            id = str(np.random.randint(10**9, 10**10))

        narrative2id[entry['narrative']] = id
        new_data[id] = {
            'narrative': entry['narrative'],
            'deception': entry['deception'],
            'stroy_length': entry['story_length'],
            'questions': []
        }

        temp_q_dict = {
            'question': entry['question'],
            'question_order': entry['question_order'],
            'choices': entry['choices'],
            'answer': entry['answer'],
        }

        if temp_q_dict not in new_data[id]['questions']:
            new_data[id]['questions'].append(temp_q_dict)

        unique_narrative.append(entry['narrative'])

visited_key = []
for key, val in data.items():
    if key in visited_key:
        print(1)
    else:
        visited_key.append(key)

