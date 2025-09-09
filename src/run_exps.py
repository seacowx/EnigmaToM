import os
import time
import argparse
import asyncio
from glob import glob
from components.utils import FileIO
from methods.tomdwm import run_dwm, ToMDwmEvaluator
from methods.timetom import run_timetom, TimeToMEvaluator
from methods.masktom import run_masktom, MaskToMEvaluator
from methods.percepttom import run_percepttom, PerceptToMEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', type=str, required=True, help='name of the dataset'
    )
    parser.add_argument(
        '--seed', type=int, default=2024, help='seed to use for the experiment'
    )
    parser.add_argument(
        '--model', type=str, required=True, help='model to use for character mask generation'
    )
    parser.add_argument(
        '--eval_method', type=str, default='masktom', help=(
            'evaluation method to use for evaluation. '
            'choose from ["masktom", "timetom", "dwm", "percepttom"]'
        )
    )
    parser.add_argument(
        '--world_state_model', type=str, default='llama8', help=(
            'world state model to use for world state generation. ' 
            'choose between ["llama8", "llama70"]'
        )
    )
    parser.add_argument(
        '--use_belief_solver', action='store_true', help='use belief solver to generate world state'
    )
    parser.add_argument(
        '--no_async', action='store_true', help='do not use async inference'
    )
    parser.add_argument(
        '--run_all', action='store_true', help='run experiments on all available settings of a dataset'
    )
    parser.add_argument(
        '--event_based', action='store_true', help='use event-based world state'
    )
    parser.add_argument(
        '--use_scene_graph', action='store_true', help='use scene graph for evaluation'
    )
    parser.add_argument(
        '--batch_inference', action='store_true', help='use OpenAI batch inference for evaluation'
    )
    return parser.parse_args()


async def main():
    args = get_args()
    file_io = FileIO()

    start_time = time.time()

    # whether to save prompt to JSONL for OpenAI batch inference
    to_jsonl = False 
    if args.batch_inference:
        to_jsonl = True

    if 'opentom' in args.data:
        data_name = 'opentom'
    elif 'tomi' in args.data:
        data_name = 'tomi'
    elif 'hitom' in args.data:
        data_name = 'hitom'
    elif 'bigtom' in args.data:
        data_name = 'bigtom'
    elif 'fantom' in args.data:
        data_name = 'fantom'
    elif 'exploretom' in args.data:
        data_name = 'exploretom'

    if not args.run_all:
        available_seeds = [args.seed]
        available_datapaths = [
            f'../data/tom_datasets/{args.data}/{args.data}_reduced_q_seed[{args.seed}].json'
        ]
    else:
        all_world_states = glob(f'../data/world_state/{args.model}/{data_name}/*.json')
        all_world_states = [ele for ele in all_world_states if 
            (data_name in ele and args.world_state_model in ele)
        ]
        available_seeds = [ele for ele in all_world_states if 'seed' in ele]
        available_seeds = [int(ele.rsplit('[', 1)[-1].rsplit(']', 1)[0]) for ele in available_seeds]
        available_seeds = list(set(available_seeds))

        available_datapaths = [
            f'../data/tom_datasets/{args.data}/{args.data}_reduced_q_seed[{seed}].json' 
            for seed in available_seeds
        ]

    # load prompt for evaluation, this prompt is independent from methods
    prompt_path = './prompts/eval_prompt_3_0.yaml'

    if args.eval_method == 'masktom':
        eval_config = file_io.load_yaml('./configs/experiment_config.yml')
        evaluator = MaskToMEvaluator(
            model_name=args.model,
            data_name=data_name,
            no_async=args.no_async,
            to_jsonl=to_jsonl,
        )

        counter = 1
        for eval_name, config in eval_config.items():
            
            if args.use_scene_graph and 'masktom' not in eval_name:
                counter += 1
                continue

            print(
                f'Running evaluation for {data_name} | {args.model} | {eval_name} | '
                f'{counter}/{len(eval_config)}'
            )

            masktom_version = config.get('masktom_version', '')
            masktom = config.get('masktom', False)

            postfix = ''
            postfix += f'_masktom_{masktom_version}' if masktom else ''
            postfix += '_event_based' if args.event_based else ''
            postfix += '_scene_graph' if args.use_scene_graph else ''

            prompt_type = config.get('prompt_type', '')
            representation = config.get('representation', '')

            for seed, data_path in zip(available_seeds, available_datapaths):
                print(f'Running evaluation for Seed {seed}')
                await run_masktom(
                    evaluator=evaluator,
                    data_path=data_path,
                    model=args.model,
                    eval_name=eval_name,
                    world_state_model=args.world_state_model,
                    prompt_path=prompt_path,
                    seed=seed,
                    no_async=args.no_async,
                    event_based=args.event_based,
                    use_scene_graph=args.use_scene_graph,
                    **config
                )
            counter += 1
            print('\n\n')

    elif args.eval_method == 'timetom':
        timetom_prompt_path = './prompts/timetom.yaml'
        evaluator = TimeToMEvaluator(
            model_name=args.model, 
            use_blief_solver=args.use_belief_solver, 
            to_jsonl=to_jsonl,
        )
        for seed, data_path in zip(available_seeds, available_datapaths):

            print(f'\n\nRunning evaluation for {data_name} | {args.model} | Seed:{seed}')
            mc_probing = True if data_name == 'hitom' else False

            await run_timetom(
                evaluator=evaluator,
                data_path=data_path,
                model=args.model,
                timetom_prompt_path=timetom_prompt_path,
                prompt_path = prompt_path,
                belief_solver=args.use_belief_solver,
                seed=seed,
                mc_probing=mc_probing,
                data_name=data_name,
            )

    elif args.eval_method == 'dwm':
        dwm_prompt_path = './prompts/dwm.yaml'
        evaluator = ToMDwmEvaluator(
            model_name=args.model,
            data_name=data_name,
        )
        for seed, data_path in zip(available_seeds, available_datapaths):
            await run_dwm(
                evaluator=evaluator,
                data_path=data_path,
                model=args.model,
                prompt_path=prompt_path,
                dwm_prompt_path=dwm_prompt_path,
                event_based=args.event_based,
                seed=seed,
            )

    elif args.eval_method == 'percepttom':
        percepttom_prompt_path = './prompts/percepttom.yaml'
        evaluator = PerceptToMEvaluator(
            model_name=args.model,
            data_name=data_name,
        )
        for seed, data_path in zip(available_seeds, available_datapaths):
            print(f'\n\nRunning evaluation for {data_name} | {args.model} | Seed:{seed} | PerceptToM')
            await run_percepttom(
                evaluator=evaluator,
                data_path=data_path,
                model=args.model,
                prompt_path=prompt_path,
                percepttom_prompt_path=percepttom_prompt_path,
                event_based=args.event_based,
                seed=seed,
            )

    end_time = time.time()

    time_interval = end_time - start_time
    if time_interval > 60:
        min_spent = f'{str(int(time_interval // 60))}min '
        sec_spent = f'{int(time_interval % 60)}sec'
    else:
        min_spent = ''
        sec_spent = f'{int(time_interval % 60)}sec'

    print(f'Finished running experiments in {min_spent}{sec_spent}')


if __name__ == '__main__':
    asyncio.run(main())
