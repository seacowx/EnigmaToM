import json
import random
import numpy as np


data = json.load(open('./opentom.json'))
data_keys = list(data.keys())
SAMPLE_SIZE = 30
random.seed(2024)
np.random.seed(2024)
sampled_keys = np.random.choice(data_keys, SAMPLE_SIZE, replace=False)
data = {key: val for (key, val) in data.items() if key in sampled_keys}

QUESTION_TEMPLATE = lambda x: f"Which sentences does {x} know about? If {x} knows about part of a sentence, copy the content of the sentence that is accessible to {x}. Select the indices to the sentences."

annotators = ['hainiu', 'siya', 'jiazheng']

counter = 0
annotator_idx = 0
unique_narrative = 0
cur_batch = []
for key, val in data.items():
    cur_narrative = val['narrative'] + ' '
    cur_narrative = cur_narrative.replace('\n\n', ' ')
    cur_narrative = cur_narrative.split('. ')
    cur_narrative = '\n'.join([f'{idx}: {x}.' for idx, x in enumerate(cur_narrative) if x != ''])

    cur_plot_info = val['plot_info']
    cur_characters = [cur_plot_info['mover'], cur_plot_info['affected_char']]

    for char in cur_characters:
        cur_question = QUESTION_TEMPLATE(char)
        prompt = f"Narrative:\n{cur_narrative}\n\n{cur_question}"

        cur_batch.append({
            'text': prompt,
            'label': ''
        })

    counter += 1
    unique_narrative += 1

    if counter and counter % 10 == 0:

        with open(f'./perspective_taking/{annotators[annotator_idx]}_data.json', 'w') as f:
            json.dump(cur_batch, f, indent=4)
        f.close()

        cur_batch = []
        annotator_idx += 1
        unique_narrative = 0
