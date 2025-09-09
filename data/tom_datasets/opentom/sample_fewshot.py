import json
import numpy as np


np.random.seed(2024)


data = json.load(open('./opentom_fewshot.json'))

for key, val in data.items():
    narrative = val['narrative']
    narrative = narrative + ' '

    narrative = narrative.replace('\n\n', '. ')
    narrative = narrative.split('. ')
    narrative = [f'{idx+1}: {narr}.' for (idx, narr) in enumerate(narrative) if narr]

    print('\n'.join(narrative))
    print('\n\n')
