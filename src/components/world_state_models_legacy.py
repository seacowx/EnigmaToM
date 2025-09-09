import gc
import re
import ast
import sys, os

sys.path.append('../src/')

import torch
from peft import PeftModel
from transformers import (
    LlamaTokenizer, 
    T5Tokenizer,
)

from rich import print
from rich.markup import escape

# from llama_recipes.inference.safety_utils import get_safety_checker
from components.utils import ModelUtils

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class WorldStateModels:

    B_INST_TOKEN, E_INST_TOKEN = '[INST]', '[/INST]'
    CACHE_DIR = '/scratch/prj/inf_llmcache/hf_cache/'

    # NOTE: the selected relations to be used with ATOMIC expert
    atomic_rel_lst = [
        # "oEffect",
        "oWant",
        "xAttr",
        "xEffect",
        "xIntent",
        "xNeed",
        "xReact",
        "Causes",
        "xReason",
        ]

    event2mind_rel_lst = [
        "Xintent",
        "Xemotion",
        "Oemotion",
    ]

    def __init__(
        self,
        base_model_name: str,
        openpi_lora_path: str='',
        atomic_lora_path: str = '',
        event2mind_lora_path: str = '',
        quantization: bool=False, 
        max_new_tokens =512, #The maximum numbers of tokens to generate
        seed: int=42, #seed value for reproducibility
        do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
        min_length: int=64, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
        use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
        top_p: float=0.9, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature: float=0.5, # [optional] The value used to modulate the next token probabilities.
        top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
        repetition_penalty: float=1.2, #The parameter for repetition penalty. 1.0 means no penalty.
        length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
        max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
        use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
        batch_size: int = 1, # The batch size to be used with the pipeline
    ):

        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample 
        self.min_length = min_length
        self.use_cache = use_cache
        self.top_p = top_p
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.max_padding_length = max_padding_length

        # Set the seeds for reproducibility
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
    
        # NOTE: the entity state model is not quantized
        print('[bold #fe8019]\nLoading world state base model...[/bold #fe8019]')
        self.base_model = ModelUtils.load_model(base_model_name, True)

        print('[bold #fe8019]\nLoading OpenPI Expert...[/bold #fe8019]', end=' ')
        # NOTE: initailize the PEFT model with OpenPI Expert
        self.peft_model, self.adapter_name_lst = ModelUtils.load_peft_model(
            self.base_model, 
            openpi_lora_path
        )
        print('[bold #fe8019]Done![/bold #fe8019]')

        print('[bold #fe8019]Loading ATOMIC Expert...[/bold #fe8019]', end=' ')
        # NOTE: Add ATOMIC & Event2Mind expert to PEFT model
        self.peft_model, self.adapter_name_lst = ModelUtils.add_lora_adapter(
            self.peft_model,
            atomic_lora_path,
            self.adapter_name_lst
        )
        print('[bold #fe8019]Done![/bold #fe8019]')

        print('[bold #fe8019]Loading Event2Mind Expert...[/bold #fe8019]', end=' ')
        self.peft_model, self.adapter_name_lst = ModelUtils.add_lora_adapter(
            self.peft_model,
            event2mind_lora_path,
            self.adapter_name_lst
        )
        print('[bold #fe8019]Done![/bold #fe8019]')

        print('[bold #b8bb26]\nLoading entity state checker...[/bold #b8bb26]')
        # self.nli_model = ModelUtils.load_t5_model('../ft_src/t5_weights/')
        
        # NOTE: Load the tokenizers
        self.world_state_tokenizer = LlamaTokenizer.from_pretrained(
            base_model_name,
            cache_dir=self.CACHE_DIR
        )
        # self.nli_tokenizer = T5Tokenizer.from_pretrained(
        #     "google/flan-t5-xl",
        #     cache_dir=self.CACHE_DIR,
        #     legacy=False,
        # )

        self.world_state_tokenizer.pad_token = self.world_state_tokenizer.eos_token

        # NOTE: specify the mask tokens used in FLAN-T5
        self.T5_SPECIAL_BOS_TOKEN = "<extra_id_0>"
        self.T5_SPECIAL_EOS_TOKEN = "<extra_id_1>"

        # self.nli_tokenizer.padding_side = "left"
        # self.nli_tokenizer.pad_token = self.nli_tokenizer.eos_token


    @staticmethod
    def openpi_prompt(narrative: str):
        dialog = {
            "role": "user",
            "content": f"Narrative: {narrative}\n\nQuestion:\nWhat are the entity state changes after the last event? Rank by importance."
        }

        return dialog

    
    def atomic_individual_prompt(self, narrative: str) -> list:

        dialog_lst = []
        for rel in self.atomic_rel_lst:
            dialog_lst.append({
                "role": "user",
                "content": f"Context: {narrative} {rel} [GEN]"
            })

        return dialog_lst
    

    def event2mind_individual_prompt(self, narrative: str) -> list:

        dialog_lst = []
        for rel in self.event2mind_rel_lst:
            dialog_lst.append({
                "role": "user",
                "content": f"Context: {narrative} {rel} [GEN]"
            })

        return dialog_lst



    def nli_prompt(self, premise: str, hypothesis: str):
        # dialog = {
        #     "role": "user",
        #     'context': f"Premise: {premise.strip()}\nHypothesis: {hypothesis.strip()}\n\nQuestion: Does the premise entail the hypothesis? Answer \"True\" if there is an entailment and \"False\" otherwise.\nAnswer: {self.T5_SPECIAL_BOS_TOKEN}",
        # }
        dialog = f"Premise: {premise.strip()}\nHypothesis: {hypothesis.strip()}\n\nQuestion: Does the premise entail the hypothesis? Answer \"True\" if there is an entailment and \"False\" otherwise.\nAnswer: {self.T5_SPECIAL_BOS_TOKEN}"
        
        return dialog


    @staticmethod
    def extract_openpi_results(entity_state: list) -> list:
        pattern1 = r"^([\w\'\s\d.]+) (is|was|are|were) ([\'\w\s]+) (before|beforehand)(?:,)?(?: and)? ([\'\w\s]+) afterwards([\.\w\s]+)?$"
        pattern2 = r"^([\w\s\d.]+) before ([\w\s]+) now ([\w\s]+)."
        re_pattern1 = re.compile(pattern1)
        re_pattern2 = re.compile(pattern2)

        all_results = []

        for entity_state_entry in entity_state:

            if (re_result1 := re_pattern1.match(entity_state_entry)):
                if len((re_groups := re_result1.groups())) >= 3:

                    re_parsed_groups = ['', '', '']
                    if '.' in re_groups[0]:
                        ent = re_groups[0].split('.')[-1].strip().lower()
                    elif re_groups[0].split()[0].isdigit():
                        ent = re_groups[0].split(' ', 1)[-1].strip().lower()
                    else:
                        ent = re_groups[0].lower().strip()

                    re_parsed_groups[0] = ent

                    group_idx = 1
                    for ele in re_groups[1:]:
                        if ele in ['is', 'was', 'are', 'were', 'before', 'beforehand', 'and']:
                            continue
                        else:
                            re_parsed_groups[group_idx] = ele
                            group_idx += 1

                        if group_idx == 3:
                            break

                    all_results.append(re_parsed_groups)

            elif (re_result2 := re_pattern2.match(entity_state_entry)):
                if len((re_groups := re_result2.groups())) == 3:
                    if '.' in re_groups[0]:
                        re_groups = [
                            re_groups[0].split('.')[-1].strip().lower(), 
                            re_groups[1].lower(), 
                            re_groups[2].lower().replace('afterwards', '').strip()
                        ]     
                    
                    else:
                        re_groups = [
                            re_groups[0].lower(), 
                            re_groups[1].lower(), 
                            re_groups[2].lower().replace('afterwards', '').strip()
                        ]     

                    all_results.append(re_groups)

        if all_results:
            return all_results
        else:
            return []


    @staticmethod
    def extract_nkg_results(atomic_results: str) -> list:
        results = atomic_results.split('\n')

        result_lst = []
        for res in results:
            try:
                result_lst.extend(ast.literal_eval(res))
            except:
                continue

        if not result_lst:
            return ['none']

        result_lst = [ele for ele in result_lst if ele != 'none']

        return result_lst

    
    @torch.no_grad()
    def openpi_generator(self, narrative_breakdown: list, narrative_sents: list):

        self.peft_model.set_adapter('openpi')

        # TODO: add support for batch processing
        results = {}
        for idx, context in enumerate(narrative_breakdown):
            prompt = self.openpi_prompt(context)

            batch = self.world_state_tokenizer(
                f"{self.world_state_tokenizer.bos_token}{self.B_INST_TOKEN} {prompt['content'].strip()} {self.E_INST_TOKEN}", 
                return_tensors="pt"
            )

            input_len = batch['input_ids'].size(-1)

            input_batch = {k: v.to("cuda") for k, v in batch.items()}

            self.peft_model.set_adapter('openpi')
            outputs = self.peft_model.generate(
                **input_batch,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                top_p=self.top_p,
                temperature=self.temperature,
                min_length=input_len + self.min_length,
                use_cache=self.use_cache,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                length_penalty=self.length_penalty,
                pad_token_id=self.world_state_tokenizer.eos_token_id,
            )

            raw_output = self.world_state_tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text = raw_output.split('[/INST]')[-1].strip()

            entity_states = output_text.split('\n')
            entity_states = self.extract_openpi_results(entity_states)

            results[f'time-stamp-{idx+1}'] = {
                'context': narrative_sents[idx],
                'cumulative_context': context,
                'original_generation': output_text,
                'entity_state_changes': entity_states,
            }

        return results


    @torch.no_grad()
    def atomic_generator(self, narrative_sents: list):

        self.peft_model.set_adapter('atomic')

        atomic_results = {}
        for idx, context in enumerate(narrative_sents):
            prompt_lst = self.atomic_individual_prompt(context)

            temp_result_dict = {}
            for (cur_rel, prompt) in zip(self.atomic_rel_lst, prompt_lst):

                batch = self.world_state_tokenizer(
                    f"{self.world_state_tokenizer.bos_token}{self.B_INST_TOKEN} {prompt['content'].strip()} {self.E_INST_TOKEN}", 
                    return_tensors="pt"
                )

                input_len = batch['input_ids'].size(-1)

                input_batch = {k: v.to("cuda") for k, v in batch.items()}

                outputs = self.peft_model.generate(
                    **input_batch,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    min_length=input_len + self.min_length,
                    use_cache=self.use_cache,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty,
                    length_penalty=self.length_penalty,
                    pad_token_id=self.world_state_tokenizer.eos_token_id,
                )

                raw_output = self.world_state_tokenizer.decode(outputs[0], skip_special_tokens=True)
                output_text = raw_output.split('[/INST]')[-1].strip()

                cur_result = self.extract_nkg_results(output_text)

                temp_result_dict[cur_rel] = cur_result

            atomic_results[f'time-stamp-{idx+1}'] = {
                'context': context,
                'original_generation': output_text,
                'atomic_graphs': temp_result_dict,
            }

        return atomic_results


    @torch.no_grad()
    def event2mind_generator(self, narrative_sents: list, route_vec: list):

        self.peft_model.set_adapter('event2mind')

        event2mind_results = {}
        for idx, (context, route_indicator) in enumerate(zip(narrative_sents, route_vec)):

            if not route_indicator:
                continue

            prompt_lst = self.event2mind_individual_prompt(context)

            temp_result_dict = {}
            for (cur_rel, prompt) in zip(self.event2mind_rel_lst, prompt_lst):

                batch = self.world_state_tokenizer(
                    f"{self.world_state_tokenizer.bos_token}{self.B_INST_TOKEN} {prompt['content'].strip()} {self.E_INST_TOKEN}", 
                    return_tensors="pt"
                )

                input_len = batch['input_ids'].size(-1)

                input_batch = {k: v.to("cuda") for k, v in batch.items()}

                self.peft_model.set_adapter('atomic')
                outputs = self.peft_model.generate(
                    **input_batch,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    min_length=input_len + self.min_length,
                    use_cache=self.use_cache,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty,
                    length_penalty=self.length_penalty,
                    pad_token_id=self.world_state_tokenizer.eos_token_id,
                )

                raw_output = self.world_state_tokenizer.decode(outputs[0], skip_special_tokens=True)
                output_text = raw_output.split('[/INST]')[-1].strip()

                # NOTE: The generation structure of event2mind is the same as that of atomic, hence use the same parser here
                cur_result = self.extract_nkg_results(output_text)

                temp_result_dict[cur_rel] = cur_result

            event2mind_results[f'time-stamp-{idx+1}'] = {
                'context': context,
                'original_generation': output_text,
                'event2mind_graphs': temp_result_dict,
            }

        return event2mind_results


    def world_state_generator(
        self, 
        narrative_breakdown: list, 
        narrative_sents: list, 
        # route_vec: list
    ) -> list:

        openpi_results = self.openpi_generator(narrative_breakdown, narrative_sents)
        # atomic_results = self.atomic_generator(narrative_sents)
        # event2mind_results = self.event2mind_generator(narrative_sents, route_vec)

        return [openpi_results] #, atomic_results, event2mind_results]


    def clear_cuda_memory(self):
        print(f'[bold #fb4934]\nClearing CUDA memory......[/bold #fb4934]', end=' ')
        del self.base_model
        del self.peft_model
        gc.collect()
        torch.cuda.empty_cache()