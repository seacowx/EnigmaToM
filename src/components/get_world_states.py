import os

import gc
import re
import ast
import json
import sys, os
from tqdm import tqdm
from abc import ABCMeta, abstractmethod

sys.path.append('../')
sys.path.append('../src/')

import torch
from peft import PeftModel
from transformers import (
    LlamaTokenizer, 
    AutoTokenizer,
    T5Tokenizer,
    TextStreamer,
    pipeline,
)

from rich import print
from rich.markup import escape

from components.utils import ModelUtils
from vllm.lora.request import LoRARequest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class WorldStateModels:

    def __init__(
        self,
        base_model_name: str,
        openpi_entity_lora_path: str='',
        openpi_entity_attr_lora_path: str='',
        world_model_name: str = 'llama',
        quantization: int=0, 
        seed: int=42, #seed value for reproducibility
        use_vllm: bool=False,
    ):
        # Set the seeds for reproducibility
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        self.CACHE_DIR = '/scratch/prj/inf_llmcache/hf_cache/'
        self.world_model_name = world_model_name
        self.use_vllm = use_vllm
    
        # NOTE: the entity state model is not quantized
        print(f'[bold #fe8019]\nBase Model: {base_model_name}[/bold #fe8019]')
        print(f'[bold #fe8019]Loading world state base model with {quantization}-bit Quantization...[/bold #fe8019]')
        self.base_model = ModelUtils.load_model(
            model_name=base_model_name, 
            quantization=quantization,
            use_vllm=use_vllm,
            enable_lora=True,
        )

        if not use_vllm:
            print('[bold #fe8019]Loading OpenPI Entity-Guided Expert...[/bold #fe8019]', end=' ')
            # NOTE: initailize the PEFT model with OpenPI Expert
            self.peft_model, self.adapter_name_lst = ModelUtils.load_peft_model(
                self.base_model, 
                openpi_entity_lora_path
            )

            print('[bold #fe8019]Loading OpenPI Entity-Attribute-Guided Expert...[/bold #fe8019]', end='')
            self.peft_model, self.adapter_name_lst = ModelUtils.add_lora_adapter(
                self.peft_model,
                openpi_entity_attr_lora_path,
                self.adapter_name_lst
            )
            print('[bold #fe8019]Done![/bold #fe8019]')
        
            # NOTE: Load the tokenizers
            self.world_state_tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                cache_dir=self.CACHE_DIR
            )

            self.world_state_tokenizer.pad_token = self.world_state_tokenizer.eos_token

            # define token for stopping the generation for OpenPI expert
            self.openpi_stop_token_lst = [
                [self.world_state_tokenizer.eos_token_id],
                self.world_state_tokenizer.encode('None', add_special_tokens=False),
                self.world_state_tokenizer.encode('none', add_special_tokens=False),
            ]
            self.openpi_stop_token_lst = sum(self.openpi_stop_token_lst, [])
        else:
            self.peft_model = self.base_model


    def set_adapter(self, adapter_name: str):
        if not self.use_vllm:
            self.peft_model.set_adapter(adapter_name)


    @staticmethod
    def openpi_entity_prompt(
        cur_context: str, 
        entity: str, 
    ) -> list:

        entity = entity.replace(':', '').strip().lower()

        dialog = {
            "role": "user",
            "content": (
                f"Event:\n{cur_context}\n\nNow, what happens to \"{entity}\"? "
                f"If \"{entity}\" undergoes state changes, list them as bullet points in the following format:\n" 
                f"- {entity.capitalize()} is now [new state].\n\nOtherwise, simply respond with \"None\"."
        )}

        return [dialog]

    @staticmethod
    def openpi_entity_attr_prompt(
        cur_context: str, 
        entity_attr: str, 
    ) -> list:

        entity_attr = entity_attr.replace(':', '').strip()

        dialog = {
            "role": "user",
            "content": (
                f"Event:\n{cur_context}\n\nNow, what happens to the \"{entity_attr}\"? " 
                f"If the {entity_attr} undergoes changes, elicit the new state as bullet points in "
                f"the following format:\n- {entity_attr.capitalize()} is now [current state].\n\n"
                "Otherwise, simply respond with \"None\"."
        )}

        return [dialog]

    @staticmethod
    def extract_openpi_entity_results(entity_state: list, entity: str) -> list:

        all_results = []

        for entity_state_entry in entity_state:

            if ' is now ' in entity_state_entry:
                entity_attr, state = entity_state_entry.rsplit(' is now ', 1)
                entity_attr = entity_attr.split('so ')[-1].split('therefore')[-1].strip()
            elif ' are now ' in entity_state_entry:
                entity_attr, state = entity_state_entry.split(' are now ')
            else:
                entity_attr, state = '', ''

            if ' of ' in entity_attr:
                entity, attr = entity_attr.split(' of ', 1)
            else:
                entity = entity_attr
                attr = ''

            state = state.replace('.', '')
            entity, attr, state = entity.strip(), attr.strip(), state.strip()

            entity = entity.replace(']', '').replace('[', '').strip()
            attr = attr.replace(']', '').replace('[', '').strip()
            state = state.replace(']', '').replace('[', '').strip()

            if not entity_attr:
                all_results.append([])
            else:
                all_results.append([attr, entity, state])

                f"This is a {all_results}"

        if all_results:
            return all_results
        else:
            return []

    
    @torch.no_grad()
    def openpi_generator(
        self, 
        narrative_breakdown_dict: dict, 
        narrative_sents_lst: list, 
        eoi_dict: dict, 
        id_lst: list,
        world_model_name: str,
        attr_guided: bool,
    ) -> dict:

        if 'llama70' in world_model_name:
            world_model_path = 'llama_70_lora'
        elif 'llama8' in world_model_name:
            world_model_path = 'llama_8_lora'

        if attr_guided:
            prompt_template = self.openpi_entity_attr_prompt
            lora_adapter_path = (
                '/scratch_tmp/prj/charnu/ft_weights/masktom/llama-8b/' 
                'openpi_entity_attribute/checkpoint-747'
            )
        else:
            prompt_template = self.openpi_entity_prompt
            lora_adapter_path = (
                '/scratch_tmp/prj/charnu/ft_weights/masktom/llama-8b/' 
                'openpi_entity/checkpoint-477'
            )

        # if not self.use_vllm:
        #
        #     world_state_data = {}
        #     for idx, (id, narrative_sents) in enumerate(tqdm(
        #             zip(id_lst, narrative_sents_lst), position=0, leave=False
        #         )):
        #
        #         narrative_breakdown = narrative_breakdown_dict[id]
        #         cur_eoi = eoi_dict[idx]
        #
        #         results = {}
        #         with tqdm(total=len(narrative_breakdown) * len(cur_eoi), position=1, leave=False) as pbar:
        #             for idx, context in enumerate(narrative_breakdown):
        #                 entity_dict = {eoi: [] for eoi in cur_eoi}
        #                 entity_original_gen = {eoi: '' for eoi in cur_eoi}
        #                 for entity in cur_eoi:
        #
        #                     prompt = prompt_template(
        #                         context,
        #                         entity,
        #                     )
        #
        #                     input_batch = self.world_state_tokenizer.apply_chat_template(
        #                         prompt,
        #                         return_tensors='pt',
        #                         tokenize = True,
        #                     ).to('cuda')
        #
        #                     outputs = self.peft_model.generate(
        #                         input_batch,
        #                         do_sample=False,
        #                         use_cache=True,
        #                         max_length=2048,
        #                         repetition_penalty=1.2,
        #                         pad_token_id=self.world_state_tokenizer.eos_token_id,
        #                         eos_token_id = self.openpi_stop_token_lst,
        #                     )
        #                     raw_output = self.world_state_tokenizer.decode(outputs[0], skip_special_tokens=True)
        #
        #                     if 'llama' in self.world_model_name:
        #                         output_text = raw_output.split('simply respond with "None".')[-1].split('assistant', 1)[-1].strip()
        #                     else:
        #                         output_text = raw_output.split('[/INST]')[-1].strip()
        #
        #                     pbar.update(1)
        #
        #                     if not output_text:
        #                         entity_original_gen[entity] = 'None'
        #                         continue
        #
        #                     entity_original_gen[entity] = output_text
        #
        #                     if 'None' in ' '.join(output_text.split()[:5]):
        #                         entity_original_gen[entity] = 'None'
        #                     else:
        #                         entity_states = output_text.split('\n')
        #                         entity_states = [ele[1:].strip() for ele in entity_states if ele.strip().startswith('-')]
        #                         entity_states = self.extract_openpi_entity_results(entity_states, entity)
        #                         entity_dict[entity] = entity_states
        #
        #                 results[f'time-stamp-{idx+1}'] = {
        #                     'original_generation': entity_original_gen,
        #                     'entity_state_changes': entity_dict,
        #                 }
        #
        #     world_state_data[id] = {
        #         'openpi_states': results,
        #     }
        #
        # else:
        prompt_lst = []
        prompt_idx_to_narrative_idx = {}
        prompt_idx_to_id = {}
        prompt_idx_dict = {}
        prompt_entity_dict = {}
        prompt_idx = 0
        for narrative_idx, (id, narrative_sents) in enumerate(tqdm(
                zip(id_lst, narrative_sents_lst), position=0, leave=False
            )):

            narrative_breakdown = narrative_breakdown_dict[id]

            cur_eoi = eoi_dict[narrative_idx]
            entity_idx = 0
            for idx, context in enumerate(narrative_breakdown):
                for entity in cur_eoi:

                    prompt = prompt_template(
                        context,
                        entity,
                    )

                    prompt_lst.append(prompt)
                    prompt_idx_dict[prompt_idx] = idx
                    prompt_entity_dict[prompt_idx] = entity
                    prompt_idx_to_narrative_idx[prompt_idx] = narrative_idx
                    prompt_idx_to_id[prompt_idx] = id
                    prompt_idx += 1
                    entity_idx += 1

        output_lst = self.peft_model.vllm_generate(
            prompts=prompt_lst,
            lora_request=LoRARequest(
                'openpi', 1, lora_adapter_path,
            ),
            temperature=0.,
        )

        world_state_data = {}
        prev_narrative_idx = 0
        prev_time_stamp = 0
        results = {}
        cur_eoi = eoi_dict[0]
        entity_dict = {eoi: [] for eoi in cur_eoi}
        entity_original_gen = {eoi: '' for eoi in cur_eoi}
        for prompt_idx, output_text in enumerate(output_lst):
            narrative_idx = prompt_idx_to_narrative_idx[prompt_idx]
            # save current results and reset for next narrative
            if narrative_idx != prev_narrative_idx:
                # add the last time-stamp results
                results[f'time-stamp-{cur_time_stamp + 1}'] = {
                    'original_generation': entity_original_gen,
                    'entity_state_changes': entity_dict,
                }

                cur_id = prompt_idx_to_id[prompt_idx-1]

                world_state_data[cur_id] = {
                    'openpi_states': results,
                }

                # reset for next narrative
                cur_eoi = eoi_dict[narrative_idx]
                entity_dict = {eoi: [] for eoi in cur_eoi}
                entity_original_gen = {eoi: '' for eoi in cur_eoi}
                prev_time_stamp = 0
                total_len = len(narrative_sents_lst[narrative_idx])
                results = {}

            cur_time_stamp = prompt_idx_dict[prompt_idx]
            entity = prompt_entity_dict[prompt_idx]
            if cur_time_stamp != prev_time_stamp:
                results[f'time-stamp-{cur_time_stamp}'] = {
                    'original_generation': entity_original_gen,
                    'entity_state_changes': entity_dict,
                }
                entity_dict = {eoi: [] for eoi in cur_eoi}
                entity_original_gen = {eoi: '' for eoi in cur_eoi}
                prev_time_stamp = cur_time_stamp

            entity_original_gen[entity] = output_text

            if 'None' in ' '.join(output_text.split()[:5]):
                entity_original_gen[entity] = 'None'
            else:
                entity_states = output_text.split('\n')
                entity_states = [
                    ele[1:].strip() for ele in entity_states if ele.strip().startswith('-')
                ]
                entity_states = self.extract_openpi_entity_results(entity_states, entity)
                entity_dict[entity] = entity_states

            prev_time_stamp = cur_time_stamp
            prev_narrative_idx = narrative_idx

        cur_id = prompt_idx_to_id[prompt_idx]
        world_state_data[cur_id] = {
            'openpi_states': results,
        }

        return world_state_data

    def world_state_generator(
        self, 
        narrative_breakdown_dict: dict, 
        narrative_sents_lst: list, 
        questions_lst: list,
        eoi_dict: dict,
        character_lst_lst: list,
        id_lst: list,
        world_model_name: str,
        attr_guided: bool,
    ) -> dict:

        world_state_data = self.openpi_generator(
            narrative_breakdown_dict, 
            narrative_sents_lst, 
            eoi_dict, 
            id_lst,
            world_model_name,
            attr_guided,
        )

        return world_state_data


class WorldStateBuilder:

    def __init__(
        self, 
        file_io, 
        world_state_model: WorldStateModels, 
        world_model_name: str
    ) -> None:
        self.file_io = file_io
        self.world_model_name = world_model_name
        self.world_state_model = world_state_model


    def build_world_state(
        self, 
        data_name: str,
        narrative_breakdown_dict: dict, 
        narrative_sents_lst: list, 
        questions_lst: list, 
        cur_eoi_dict: list,
        character_lst_lst: list,
        id_lst: list,
        world_model_name: str,
        attr_guided: bool,
    ) -> list | dict:
        
        eoi_dict = {}
        for idx, (questions, character_lst) in enumerate(zip(questions_lst, character_lst_lst)):

            # reduce eoi for HiToM to improve performance
            if data_name == 'hitom' and world_model_name == 'llama70':
                cur_eoi = cur_eoi_dict[idx]['response'][:2]
            else:
                cur_eoi = cur_eoi_dict[idx]['response'][:5]

            # NOTE: character EOI is probably not needed
            # the main purpose of character-based EOI is to conduct perspective-taking, which is taken care of by
            # the scene graph
            #
            # cur_eoi = [ele for ele in cur_eoi if not any(char in ele for char in character_lst)]
            #
            # if attr_guided:
            #     character_attr = ['awareness', 'feeling']
            #     character_lst = [ele for ele in character_lst if any(ele in q for q in questions)]
            #     character_eoi = [
            #         f"{attr} of {char}" for char in character_lst for attr in character_attr
            #     ]
            #     cur_eoi = cur_eoi + character_eoi
            # else:
            #     cur_eoi = cur_eoi + character_lst

            eoi_dict[idx] = cur_eoi

        world_state_data = self.world_state_model.world_state_generator(
            narrative_breakdown_dict,
            narrative_sents_lst,
            questions_lst,
            eoi_dict,
            character_lst_lst,
            id_lst,
            world_model_name,
            attr_guided,
        )

        return world_state_data
