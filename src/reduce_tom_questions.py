import argparse
from tqdm import tqdm

import numpy as np

from components.utils import FileIO
from generate_states.generate_reduced_questions import reduce_tom_questions


def get_args():
    parser = argparse.ArgumentParser()
    # TODO: add argument to specify the location of openai config file
    parser.add_argument(
        '--data_path', type=str, required=True, help='path to the dataset'
    )
    parser.add_argument(
        '--prompt_path', type=str, default='./prompts/prompt_8_0.yaml', help='path to the prompt bank file'
    )
    return parser.parse_args()


def main():
    args = get_args()
    file_io = FileIO()

    data_name = args.data_path.split('/')[-1].split('.')[0].split('_')[0]
    data = file_io.load_json(args.data_path)
    data_keys = list(data.keys())
    
    try:
        data_character_dict = file_io.load_json(f'../data/masktom/ner/{data_name}_char_name.json')
    except:
        raise ValueError(f"Character names for {data_name} not found. Please run NER to generate character names.")

    #FIXME: adjust sample size when running eval
    SAMPLE_SIZE = 100
    np.random.seed(2024)
    sampled_keys = np.random.choice(data_keys, SAMPLE_SIZE, replace=False)
    sampled_data = {key: val for (key, val) in data.items() if key in sampled_keys}

    sampled_data = reduce_tom_questions(
        sampled_data,
        data_name,
        args.prompt_path,
        data_character_dict,
    )

    file_io.save_json(sampled_data, f'../data/tom_datasets/{data_name}/{data_name}_reduced_q.json')


if __name__ == '__main__':
    main()

