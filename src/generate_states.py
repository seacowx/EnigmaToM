import os
import argparse
import requests

import asyncio
import numpy as np
from rich import print

from components.utils import FileIO
from components.llms import OpenAIInference, OpenAIAsyncInference

from components.lora_utils import load_lora_model
from components.get_world_states import WorldStateModels

from components.get_events import extract_events

from generate_states.generate_locations import identify_locations
from generate_states.generate_ner_results import generate_ner_results
from generate_states.generate_world_state import generate_world_state
from generate_states.generate_scene_graph import generate_scene_graph
from generate_states.generate_reduced_questions import reduce_tom_questions
from generate_states.generate_character_state import generate_character_state
from generate_states.generate_entity_of_interest import generate_entity_of_interest
from generate_states.generate_narrative_locations import generate_narrative_location

# setting logging level for HuggingFace Transformers
import warnings
from transformers.utils import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def load_model(
    model_name:str, 
    model_path:str = '',
) -> tuple:

    if 'gpt' in model_name:
        model_kwargs = file_io.load_yaml(os.path.expanduser('~/openai_info.yml'))
    else:
        vllm_config = file_io.load_yaml('./configs/vllm_configs.yml')
        cur_model_config = vllm_config[model_name]

        model_kwargs = dict(
            base_url=cur_model_config['base_url'],
            api_key=cur_model_config['api_key']
        )

    model = OpenAIInference(
        **model_kwargs,
    )

    async_model = OpenAIAsyncInference(
        **model_kwargs,
    )

    if 'gpt' not in model_name:
        try:
            attemp = requests.get(cur_model_config['base_url'])
        except:
            raise Exception(
                f'Initiate server at {cur_model_config["script_path"]}'
                'before running the script.'
            )

        if attemp.status_code != 401:
            raise Exception(
                f'Initiate server at {cur_model_config["script_path"]}'
                'before running the script.'
            )

    
    return model, async_model


async def produce_components(
    data_name, 
    sampled_data, 
    augmented_data, 
    generation_config: dict, 
    seed: int,
    event_based: bool,
    to_jsonl: bool,
):
    if not to_jsonl:
        model, async_model = load_model(
            model_name=args.model,
        )
    else:
        model, async_model = None, None

    postfix = ''
    postfix += '_event' if event_based else ''

    if not os.path.exists(
        f'../data/masktom/locations/{args.model}/'
        f'{data_name}{postfix}_samples_seed[{seed}].json'
    ):

        print(
            f'[bold #b8bb26]\n'
            f'Detecting locations for {data_name.upper()}...'
            '[/bold #b8bb26]'
        )
        await identify_locations(
            data=sampled_data, 
            augmented_data=augmented_data,
            data_name=data_name,
            model=model,
            async_model=async_model,
            model_name=args.model,
            prompt_path=args.prompt_path, 
            generation_config=generation_config,
            seed=seed,
            event_based=event_based,
            to_jsonl=args.to_jsonl,
        )

    if not os.path.exists(
        f'../data/masktom/character_state/{args.model}/'
        f'{data_name}{postfix}_samples_seed[{seed}].json'
    ):

        print(
            f'[bold #b8bb26]\n'
            f'Generating character state for {data_name.upper()}...'
            '[/bold #b8bb26]'
        )
        await generate_character_state(
            data=sampled_data, 
            augmented_data=augmented_data,
            data_name=data_name, 
            model=model,
            async_model=async_model,
            model_name=args.model,
            prompt_path=args.prompt_path, 
            generation_config=generation_config, 
            seed=seed,
            event_based=event_based,
            to_jsonl=args.to_jsonl,
        )

    if not os.path.exists(
        f'../data/masktom/narrative_loc/{args.model}/'
        f'{data_name}{postfix}_samples_seed[{seed}].json'
    ):

        print(
            f'[bold #b8bb26]\n'
            f'Generating narrative locations for {data_name.upper()}...'
            '[/bold #b8bb26]'
        )
        await generate_narrative_location(
            data=sampled_data,
            augmented_data=augmented_data,
            data_name=data_name, 
            model=model,
            async_model=async_model,
            model_name=args.model,
            prompt_path=args.prompt_path, 
            seed=seed,
            event_based=event_based,
            to_jsonl=args.to_jsonl,
        )

    if not os.path.exists(
        f'../data/masktom/eoi/{args.model}/{data_name}{postfix}_eoi_seed[{seed}].json'
    ):
        print(
            f'[bold #b8bb26]\n'
            f'Generating entities-of-interest without attribute for {data_name.upper()}...'
            '[/bold #b8bb26]'
        )
        await generate_entity_of_interest(
            data=sampled_data, 
            augmented_data=augmented_data,
            data_name=data_name, 
            model=model,
            async_model=async_model,
            model_name=args.model,
            prompt_path=args.prompt_path, 
            add_attr=False,
            seed=seed,
            event_based=event_based,
            to_jsonl=args.to_jsonl,
        )
        print(f'[bold #b8bb26]DONE!\n[/bold #b8bb26]')

    if not os.path.exists(
        f'../data/masktom/eoi/{args.model}/{data_name}{postfix}_attr_eoi_seed[{seed}].json'
    ):
        print(
            f'[bold #b8bb26]\n'
            f'Generating entities-of-interest with attribute for {data_name.upper()}...'
            '[/bold #b8bb26]'
        )
        await generate_entity_of_interest(
            data=sampled_data, 
            augmented_data=augmented_data,
            data_name=data_name, 
            model=model,
            async_model=async_model,
            model_name=args.model,
            prompt_path=args.prompt_path, 
            add_attr=True,
            seed=seed,
            event_based=event_based,
            to_jsonl=args.to_jsonl,
        )
        print(f'[bold #b8bb26]DONE!\n[/bold #b8bb26]')


def produce_world_state(
    data_name: str, 
    sampled_data: dict, 
    augmented_data: dict,
    world_model_name: str,
    seed: int,
    attr_guided: bool,
    event_based: bool,
    world_state_model: WorldStateModels | None,
):

    postfix = ''
    postfix += '_event' if event_based else ''

    # check if world state has been generated
    if os.path.exists(
        f'../data/world_state/{args.model}/{data_name}/'
        f'{world_model_name}{postfix}_examples_seed[{seed}].json'
    ):
        print(
                f'[bold #fb4934]World states already generated for'
                f'{data_name.upper()} {args.model}. Exiting...[/bold #fb4934]'
            )
        raise SystemExit()
    else:
        print(
            f'\n\n[bold #b8bb26]'
            f'Generating world states for {data_name.upper()} with {args.model}...'
            '[/bold #b8bb26]'
        )

    # WARNING: this is not neded if using vllm for inference

    # if attr_guided:
    #     print('[bold #fe8019]Activating OpenPI Entity-Attribute-Guided expert...[/bold #fe8019]')
    #     world_state_model.set_adapter('openpi_entity_attr') 
    # else:
    #     print('[bold #fe8019]Activating OpenPI Entity-Guided expert...[/bold #fe8019]')
    #     world_state_model.set_adapter('openpi_entity_attr') 

    # generate entity-guided world state
    print(f'[bold #b8bb26]\nGenerating World States for {data_name.upper()}...[/bold #b8bb26]')
    generate_world_state(
        world_state_model=world_state_model,
        sampled_data=sampled_data,
        augmented_data=augmented_data,
        data_name=data_name,
        model_name=args.model,
        world_model_name=world_model_name,
        attr_guided=attr_guided,
        seed=seed,
        event_based=event_based,
    )
    print(f'[bold #b8bb26]DONE!\n\n[/bold #b8bb26]')



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data', 
        type=str, 
        required=True, 
        help='name of the dataset'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='gpt', 
        help='model to use for character mask generation'
    )
    parser.add_argument(
        '--prompt_path', 
        type=str, 
        default='./prompts/prompt_9_0.yaml', 
        help='path to the prompt bank file'
    )
    parser.add_argument(
        '--world_state', 
        action='store_true', 
        help='generate world states'
    )
    parser.add_argument(
        '--world_state_model', 
        type=str, 
        default='llama8', 
        help='world state model to use. choose between "llama8" and "llama70"'
    )
    parser.add_argument(
        '--use_vllm', 
        action='store_true', 
        help='use vLLM for world state generation'
    )
    parser.add_argument(
        '--attr_guided', 
        action='store_true', 
        help='use attribute-guided world state generation'
    )
    parser.add_argument(
        '--no_async', 
        action='store_true', 
        help='run the code synchronously'
    )
    parser.add_argument(
        '--event_based', 
        action='store_true', 
        help='use event-based world state generation'
    )
    parser.add_argument(
        '--scene_graph', 
        action='store_true', 
        help='generate scene graph'
    )
    parser.add_argument(
        '--to_jsonl',
        action='store_true',
        help='save the results in JSONL format'
    )
    parser.add_argument(
        '--extract_events', action='store_true', help='extract events'
    )

    return parser.parse_args()


async def main():
    global args, file_io
    args = get_args()
    file_io = FileIO()

    data_name = args.data
    data_fpath = f'../data/tom_datasets/{data_name}/{data_name}.json'
    data = file_io.load_json(data_fpath)
    data_keys = list(data.keys())

    if 'fantom' in args.data or 'exploretom' in args.data:
        SAMPLE_SIZE = 50
    else:
        SAMPLE_SIZE = 100

    # initialize entity state model 
    world_state_model = None
    if args.world_state or args.scene_graph:
        world_state_model = load_lora_model(
            world_model_name=args.world_state_model,
            use_vllm=args.use_vllm,
        )

    AVAILABLE_SEEDS = [12, 42, 96, 2012, 2024]

    for seed in AVAILABLE_SEEDS:
        event_verbose = ' | EVENT-BASED' if args.event_based else ''
        print(
            f'[bold #fb4934]'
            f'Processing {data_name.upper()} | {args.model}{event_verbose} | SEED {seed}...'
            f'[/bold #fb4934]'
        )
        np.random.seed(seed)
        sampled_keys = np.random.choice(data_keys, SAMPLE_SIZE, replace=False)
        sampled_data = {key: val for (key, val) in data.items() if key in sampled_keys}

        ner_fpath = f'../data/masktom/ner/{data_name}_char_name_seed[{seed}].json'
        if not os.path.exists(ner_fpath):
            print(
                f'[bold #fb4934]'
                f'NER vectors missing for {data_name.upper()} | SEED {seed}. '
                'Initiating NER model...'
                '[/bold #fb4934]'
            )

            generate_ner_results(
                data=sampled_data,
                data_name=data_name,
                seed=seed,
            )
            print(f'[bold #fb4934]Done![/bold #fb4934]')
            continue

        character_dict = file_io.load_json(ner_fpath)

        prompt_dict = file_io.load_yaml(args.prompt_path)

        augment_fpath = f'../data/augmented_data/{args.model}/{data_name}_augmented_seed[{seed}].json'
        if args.extract_events:
            if not os.path.exists(augment_fpath) and 'gpt' not in args.model:
                print(f'[bold #b8bb26]Extracting events for {data_name.upper()}...[/bold #b8bb26]')
                augmented_data = await extract_events(
                    data=sampled_data,
                    character_dict=character_dict,
                    data_name=data_name,
                    model_name=args.model,
                    prompt_dict=prompt_dict,
                    no_async=args.no_async,
                    seed=seed,
                )

        elif 'gpt' not in args.model:
            augmented_data = file_io.load_json(augment_fpath)
        else:
            augmented_data = {}

        reduce_fpath = f'../data/tom_datasets/{data_name}/{data_name}_reduced_q_seed[{seed}].json'
        if not os.path.exists(reduce_fpath):
            print(f'[bold #b8bb26]Reducing ToM questions for {data_name.upper()}...[/bold #b8bb26]')
            reduced_data = await reduce_tom_questions(
                data=sampled_data,
                data_name=data_name,
                model_name=args.model,
                prompt_path=args.prompt_path,
                data_character_dict=character_dict,
            )
            file_io.save_json(reduced_data, reduce_fpath)
        else:
            reduced_data = file_io.load_json(reduce_fpath)

        # load LLM-related configs
        generation_config = file_io.load_yaml('./configs/generation_config.yml')
        model_info = {}
        if 'gpt' not in args.model:
            model_info = file_io.load_yaml('./configs/model_info.yml')
            model_info = model_info.get(args.model, 0)

            assert model_info, f'Model {args.model} not supported. Please choose from {model_info.keys()}'

        if args.world_state:
            produce_world_state(
                data_name=data_name, 
                sampled_data=sampled_data, 
                augmented_data=augmented_data,
                world_model_name=args.world_state_model,
                seed=seed,
                attr_guided=args.attr_guided,
                event_based=args.event_based,
                world_state_model=world_state_model,
            )
        elif args.scene_graph:
            generate_scene_graph(
                data_name=data_name,
                sampled_data=sampled_data, 
                augmented_data=augmented_data,
                world_model_name=args.world_state_model,
                world_state_model=world_state_model,
                seed=seed,
                model_name=args.model,
                use_vllm=args.use_vllm,
                event_based=args.event_based,
            )
        else:
            await produce_components(
                data_name=data_name,
                sampled_data=sampled_data,
                augmented_data=augmented_data,
                generation_config=generation_config,
                seed=seed,
                event_based=args.event_based,
                to_jsonl=args.to_jsonl,
            )


if __name__ == '__main__':
    asyncio.run(main())

