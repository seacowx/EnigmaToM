import os
import requests
import json, yaml, pickle, csv
from typing import Iterable, Tuple

import gc
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    T5ForConditionalGeneration,
)

from components.llms import OpenAIInference, OpenAIAsyncInference, vLLMInference

# import for progress bar
from rich import print
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    SpinnerColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

class FileIO:

    @staticmethod
    def load_json(fpath: str):
        with open(fpath, 'r') as f:
            return json.load(f)


    @staticmethod
    def save_json(obj, fpath: str):
        with open(fpath, 'w') as f:
            json.dump(obj, f, indent=4)
        f.close()


    @staticmethod
    def save_jsonl(obj, fpath: str):
        with open(fpath, 'w') as f:
            for item in obj:
                f.write(json.dumps(item) + '\n')


    @staticmethod
    def load_txt(fpath: str):
        with open(fpath, 'r') as f:
            return f.read().splitlines()


    @staticmethod
    def load_csv(fpath: str):
        out_data = []
        with open(fpath, 'r') as fd:
            rd = csv.reader(fd, delimiter=",", quotechar='"')
            for row in rd:
                out_data.append(row)
        return out_data


    @staticmethod
    def load_jsonl(fpath: str):
        with open(fpath, 'r') as f:
            return [json.loads(line) for line in f.read().splitlines()]


    @staticmethod
    def load_yaml(fpath: str):
        with open(fpath, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)


    @staticmethod
    def save_pickle(obj, fpath: str):
        with open(fpath, 'wb') as f:
            pickle.dump(obj, f)
        f.close()

    
    @staticmethod
    def save_py(code: str, fpath: str):
        with open(fpath, 'w') as f:
            f.write(code)
        f.close()


    @staticmethod
    def load_pickle(fpath: str):
        with open(fpath, 'rb') as f:
            return pickle.load(f)


    @staticmethod
    def load_tsv(fpath: str):
        out_data = []
        with open(fpath, 'r') as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                out_data.append(row)
        return out_data


class ModelUtils:

    @staticmethod
    def load_model(model_name, quantization: int, use_vllm: bool, enable_lora: bool):

        if use_vllm:
            if quantization < 16:
                is_quantized = True
            else:
                is_quantized = False

            model = vLLMInference(
                model_name=model_name,
                quantization=is_quantized,
                enable_lora=enable_lora,
                gpu_memory_utilization=0.9,
            )
        else:
            if quantization and quantization != 32:
                bnb_config = None
                if quantization == 4:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                
                elif quantization == 8:
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold = 6.,
                    )

                elif quantization == 16:
                    dtype=torch.float16

                if bnb_config:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        return_dict=True,
                        quantization_config=bnb_config,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                    )

                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        return_dict=True,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        torch_dtype=dtype,
                    )

            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    return_dict=True,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )

        return model


    # Function to load the PeftModel for performance optimization
    @staticmethod
    def load_peft_model(model, lora_path: str) -> Tuple[PeftModel, list]:
        if "entity" in lora_path:
            adapter_name = "openpi_entity" 
        elif "entity_attr" in lora_path:
            adapter_name = "openpi_entity_attr"
        elif "event2mind" in lora_path:
            adapter_name = "event2mind"

        peft_model = PeftModel.from_pretrained(
            model=model, 
            model_id=lora_path,
            adapter_name=adapter_name,
        )

        adapter_name_lst = [adapter_name]
        return peft_model, adapter_name_lst


    @staticmethod
    def add_lora_adapter(
        peft_model: PeftModel, 
        lora_path: str, 
        adapter_name_lst: list
    ) -> Tuple[PeftModel, list]:

        if "entity_attr" in lora_path:
            adapter_name = "openpi_entity_attr"
        elif "entity" in lora_path:
            adapter_name = "openpi_entity"
        elif "openpi" in lora_path:
            adapter_name = "openpi" 
        elif "event2mind" in lora_path:
            adapter_name = "event2mind"

        peft_model.load_adapter(
            model_id = lora_path,
            adapter_name = adapter_name
        )

        adapter_name_lst.append(adapter_name)

        return peft_model, adapter_name_lst

    
    @staticmethod 
    def set_active_adapter(peft_model: PeftModel, adapter_name: str) -> PeftModel:
        return peft_model.set_adapter(adapter_name)


    @staticmethod
    def load_t5_model(model_path):
        model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            cache_dir='/scratch/prj/inf_llmcache/hf_cache/',
            device_map='auto',
        )
        return model


class AtomicHumanReadable():
    """
    Human-readable relations for augmenting entries
    Acquired from the original ATOMIC paper: Table 9
    """
    rel_map = {
        "atlocation": "located or found at/in/on",
        "capableof": "is/are capable of",
        "causes": "causes",
        "causesdesire": "makes someone want",
        "createdby": "is created by",
        "desiresr": "desires",
        "hasa": "has, possesses or contains",
        "hasfirstsubevent": "BEGINS with the event/action",
        "haslastsubevent": "ENDS with the event/action",
        "hasprerequisite": "to do this, one requires",
        "hasproperty": "can be characterized by being/having",
        "hassubevent": "includes the event/action",
        "hinderedby": "can be hindered by",
        "instanceof": "is an example/instance of",
        "isafter": "happens after",
        "isbefore": "happens before",
        "isfilledby": "blank can be filled by",
        "madeof": "is made of",
        "madeupof": "made (up) of",
        "motivatedbygoal": "is a step towards accomplishing the goal",
        "notdesires": "do(es) NOT desire",
        "objectuse": "used for ",
        "usedfor": "used for ",
        "oeffect": "as a result, Y or others will",
        "oreact": "as a result, Y or others feels",
        "owant": "as a result, Y or others want",
        "partof": "is a part of",
        "receivesaction": "can receive or be affected by the action",
        "xattr": "X is seen as",
        "xeffect": "as a result, PersonX will",
        "xintent": "because PersonX wanted",
        "xneed": "but before, PersonX needed",
        "xreact": "as a result, PersonX feels",
        "xreason": "because",
        "xwant": "as a result, PersonX wants",
    }

    def get_human_readable(self, rel: str):
        return self.rel_map.get(rel, '[UNK]')


class Event2MindHumanReadable():
    """
    Human-readable relations for augmenting entries
    Acquired from the original ATOMIC paper: Table 9
    """
    rel_map = {
        'xintent': 'PersonX intends to',
        'xemotion': 'PersonX feels', 
        'oemotion': 'PersonX makes others feel'
    }

    def get_human_readable(self, rel: str):
        return self.rel_map.get(rel, '[UNK]')


def seacow_progress(item: Iterable, message: str, color: str):

    progress = Progress(
        SpinnerColumn(),
        TextColumn(f"[{color}]{message}"),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    )

    with progress:
        for n in progress.track(item):
            yield n

        
def clear_cuda_memory(model_var_lst: list):
    """
    Clear CUDA memory for HuggingFace models
    """
    for model in model_var_lst:
        print(f'[bold #fb4934]\nClearing CUDA memory...')
        del model
    gc.collect()
    torch.cuda.empty_cache()


def load_model(
    model_name:str, 
    no_async: bool,
    model_path:str = '',
) -> OpenAIInference | OpenAIAsyncInference:

    file_io = FileIO()

    # code for inferencing with OpenAI-campatible server (slow!)
    vllm_config = file_io.load_yaml('./configs/vllm_configs.yml')
    cur_model_config = vllm_config[model_name]

    try:
        attemp = requests.get(cur_model_config['base_url'])
    except:
        raise Exception(f'Initiate server at {cur_model_config["script_path"]} before running the script.')

    if attemp.status_code != 401:
        raise Exception(f'Initiate server at {cur_model_config["script_path"]} before running the script.')

    if no_async:
        model = OpenAIInference(
            base_url=cur_model_config['base_url'],
            api_key=cur_model_config['api_key'],
        )
    else:
        model = OpenAIAsyncInference(
            base_url=cur_model_config['base_url'],
            api_key=cur_model_config['api_key'],
        )

    # model = vLLMInference(
    #     model_name=model_name,
    #     download_dir=model_path,
    # )
    
    return model
