import gc
from rich import print

import torch
from transformers import pipeline, logging
from transformers import AutoTokenizer, AutoModelForTokenClassification

from components.utils import seacow_progress


logging.set_verbosity_error()


class NERModule:

    def __init__(self):

        print('[bold #fabd2f]\nLoading NER Module...[/bold #fabd2f]')

        self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "dslim/bert-large-NER"
        )

        self.ner_model = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=0)


    @staticmethod
    def organize_ner_result(ner_result: list, sent: str, n_tokens: list) -> list:

        character_lst = []
        prev_token = ''
        prev_end_idx, cur_start_idx = -1, 0
        for out_dict in ner_result:
            cur_token = out_dict['word']
            entity_type = out_dict['entity']

            if 'PER' in entity_type or '##' in cur_token:

                # the token is a non-head subword
                if '##' in cur_token:
                    cur_start_idx = out_dict['start']
                    if cur_start_idx == prev_end_idx:
                        prev_token += cur_token.replace('##', '')
                    else:
                        character_lst.append(prev_token)
                        prev_token = ''

                    prev_end_idx = out_dict['end']

                # the token is a head subword
                else:
                    prev_token = cur_token
                    prev_end_idx = out_dict['end']

            else:
                # if the token is not a PERSON, add current accumulated prev_token and reset
                character_lst.append(prev_token)
                prev_token = ''
                prev_end_idx = -1

            # if entity_type == 'B-PER':
            #     if prev_token:
            #         character_lst.append(prev_token)

            #     prev_end_idx = out_dict['end'] 
            #     prev_token = cur_token

            # elif entity_type == 'I-PER':
            #     cur_start_idx = out_dict['start']
            #     if cur_start_idx == prev_end_idx:
            #         prev_token += cur_token.replace('##', '')
            #     else:
            #         prev_token = ''

        character_lst.append(prev_token)
        character_lst = list(set([ele for ele in character_lst if ele.lower() in n_tokens]))

        return character_lst

    
    def ner_inference(self, narrative_sents: dict, data_name: str) -> tuple:
        person_dict = {}
        route_dict = {}

        for key, n_sents in seacow_progress(narrative_sents.items(), "Generating routing vectors", "#b8bb26"):

            temp_route_vec = []
            if data_name == 'fantom':
                char_res = list(set([ele.split(':', 1)[0].strip() for ele in n_sents]))
            else:
                n_tokens = [ele.split() for ele in n_sents]
                n_tokens = list(set(sum(n_tokens, [])))
                n_tokens = [ele.lower().strip() for ele in n_tokens]

                ner_result = self.ner_model(n_sents, batch_size=len(n_sents))

                char_res = [
                    self.organize_ner_result(ele, n_sents[idx], n_tokens) 
                    for (idx, ele) in enumerate(ner_result)
                ]

                char_res = list(set(sum(char_res, [])))
                for i in range(len(char_res)):
                    cur_char_name = char_res[i]
                    other_char_names = char_res[:i] + char_res[i+1:]
                    if any(cur_char_name in other_char_name for other_char_name in other_char_names):
                        char_res[i] = ''

                char_res = [ele for ele in char_res if ele]

                for out in ner_result:
                    if out is not None:  # Add condition to check if out is not None
                        out = [ele['entity'].lower() for ele in out]
                        temp_route_vec.append(1 if any('per' in r for r in out) else 0)
                    else:
                        temp_route_vec.append(0)

            route_dict[key] = temp_route_vec
            person_dict[key] = char_res

        return route_dict, person_dict


    def clear_cuda_memory(self):
        print(f'[bold #fb4934]\nClearing CUDA memory......[/bold #fb4934]', end=' ')
        del self.model
        del self.ner_model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

            