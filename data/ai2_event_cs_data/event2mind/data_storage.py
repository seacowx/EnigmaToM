import numpy as np
from typing import List, Dict

class HumanReadable():
    """
    Human-readable relations for augmenting entries
    Acquired from the original ATOMIC paper: Table 9
    """
    rel_map = {
        'Xintent': 'PersonX intends to',
        'Xemotion': 'PersonX feels', 
        'Oemotion': 'PersonX makes others feel'
    }

    def get_human_readable(self, rel: str):
        return self.rel_map.get(rel, '[UNK]')


def augmentation_prompt(
    head: str | List[str], 
    rel: str | List[str], 
    tail: List[str] | List[List[str]]
) -> str | List[Dict[str, str]]:

    """
    Unformated prompt used for augmenting the events 
    """

    if isinstance(head, str):
        prompt = f"""Corrupted Event Description: {head}
Event Effect: {rel.capitalize()} {np.random.choice(tail)}

According to the corrupted event and its effect, what are the 5 most likely words to fill in the blank (___)? Provide the 5 most likely words in a list. Do not give any explanation or additional information. Answer in the following format:
The most likely 5 words at the blank are: [word1, word2, word3, word4, word5]"""

        prompt_lst = [{
            "role": "user", 
            "content": prompt,
        }]

    else:
        prompt_lst = []
        for (h, r, t) in zip(head, rel, tail):
            prompt = f"""Corrupted Event Description: {h}
Event Effect: {r.capitalize()} {np.random.choice(t)}

According to the corrupted event and its effect, what are the 5 most likely words to fill in the blank (___)? Provide the 5 most likely words in a list. Do not give any explanation or additional information. Answer in the following format:
The most likely 5 words at the blank are: [word1, word2, word3, word4, word5]"""
            prompt_lst.append([{
                'role': 'user',
                'content': prompt,
            }])

    return prompt_lst
