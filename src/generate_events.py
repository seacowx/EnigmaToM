import re
import os, sys
import argparse
import requests
from tqdm import tqdm

import numpy as np

from components.utils import FileIO
from components.llms import OpenAIInference

from components.get_events import extract_events


def load_model(
    model_name:str, 
) -> OpenAIInference:

    vllm_config = file_io.load_yaml('./configs/vllm_configs.yml')
    cur_model_config = vllm_config[model_name]

    try:
        attemp = requests.get(cur_model_config['base_url'])
    except:
        raise Exception(f'Initiate server at {cur_model_config["script_path"]} before running the script.')

    if attemp.status_code != 401:
        raise Exception(f'Initiate server at {cur_model_config["script_path"]} before running the script.')

    model = OpenAIInference()
    model.init_model(
        base_url=cur_model_config['base_url'],
        api_key=cur_model_config['api_key'],
    )
    
    return model


def get_args():
    parser = argparse.ArgumentParser(description='Generate events from world states')
    parser.add_argument(
        '--data_path', type=str, required=True, help='Path to the ToM data file'
    )
    parser.add_argument(
        '--model', type=str, required=True, help='Model to use for event generation'
    )
    parser.add_argument(
        '--prompt_path', type=str, default='./prompts/prompt_9_0.yaml', help='Path to the prompt file'
    )
    return parser.parse_args()


def main():
    global args, file_io
    args = get_args()
    file_io = FileIO()

    prompt_dict = file_io.load_yaml(args.prompt_path)

    data_name = args.data_path.split('/')[-1].split('.')[0].split('_')[0]
    data = file_io.load_json(args.data_path)
    data_keys = list(data.keys())

    character_dict = file_io.load_json(f'../data/masktom/ner/{data_name}_char_name.json')

    #FIXME: adjust sample size when running eval
    SAMPLE_SIZE = 100
    np.random.seed(2024)
    sampled_keys = np.random.choice(data_keys, SAMPLE_SIZE, replace=False)
    sampled_data = {key: val for (key, val) in data.items() if key in sampled_keys}

    model = load_model(model_name=args.model)

    extract_events(
        data=sampled_data,
        character_dict=character_dict,
        data_name=data_name,
        model=model,
        model_name=args.model,
        prompt_dict=prompt_dict,
    )


if __name__ == '__main__':
    main()




