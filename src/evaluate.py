import os
import argparse
from tqdm import tqdm
from glob import glob

from eval_utils import Evaluator
from components.utils import FileIO

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import csv
import pandas as pd
from rich import box
from rich.table import Table
from rich.console import Console


def make_table(sample_size: int, seed_lst: list, world_state_model: str):
    table = Table(
        title=f"Evaluation Results for {args.model.upper()}\n"
            f"Sample Size: {sample_size} | {' '.join([str(ele) for ele in seed_lst])}\n"
            f"World Model: {world_state_model.capitalize()}"
        # box=box.MINIMAL_DOUBLE_HEAD,
    )
    table.add_column("Framework", justify="center", no_wrap=True)
    table.add_column("Version", justify="center")
    table.add_column("WM", justify="center")
    table.add_column("Event", justify="center")
    table.add_column("Attr", justify="center")
    table.add_column("Prompt Format", justify="center")
    table.add_column("Scene Graph", justify="center")
    table.add_column("Acc", justify="center", style="green")
    # table.add_column("F1", justify="center", style="green")
    table.add_column("∆Acc", justify="center", style="#fb4934")
    # table.add_column("∆F1", justify="center", style="#fb4934")
    return table


def save_results_to_csv(
    output_path: str,
    timetom_scores, 
    vanilla_scores, 
    masktom_scores, 
):

    OUTPUT_ROOT_PATH = '../data/evaluation/'

    # Define headers based on the data structure
    headers = [
        'Method',
        'Version', 
        'World_Model',
        'Event_Based',
        'Attr_Guided',
        'Format',
        'Scene_Graph',
        'μAcc',
        'μF1',
        'σAcc',
        'σF1',
    ]
    
    # Combine all scores
    vanilla_results = []
    for postfix in vanilla_scores:
        vanilla_results.extend([vanilla_scores[postfix]])

    vanilla_df = pd.DataFrame(vanilla_results, columns=headers)
    vanilla_df.drop(columns=['μF1', 'σF1'], inplace=True)

    timetom_results = []
    for postfix in timetom_scores:
        timetom_results.extend([timetom_scores[postfix]])

    timetom_df = pd.DataFrame(timetom_results, columns=headers)
    timetom_df.drop(columns=['μF1', 'σF1'], inplace=True)

    masktom_results = []
    for postfix in masktom_scores:
        masktom_results.extend([masktom_scores[postfix]])

    masktom_df = pd.DataFrame(masktom_results, columns=headers)
    masktom_df.drop(columns=['μF1', 'σF1'], inplace=True)
    masktom_df = masktom_df.sort_values(
        by=['Version', 'Format', 'Scene_Graph'],
        ascending=[True, True, True]
    )

    out_df = pd.concat([vanilla_df, timetom_df, masktom_df], ignore_index=True)
    out_df.round(3)
    
    # Write to CSV
    output_file_path = os.path.join(OUTPUT_ROOT_PATH, output_path)
    out_df.to_csv(output_file_path, index=False)
    
    print(f"Results saved to {output_file_path}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, default='llama3-8b', help='Model to evaluate'
    )
    parser.add_argument(
        '--data', type=str, default='tomi', help='Dataset to evaluate'
    )
    parser.add_argument(
        '--seed', type=int, default=0, help='Seed to be evaluated'
    )
    parser.add_argument(
        '--evaluate_all', action='store_true', help='Evaluate all seeds of a dataset+model'
    )
    parser.add_argument(
        '--world_state_model', type=str, default='llama8', help=(
            'World state model to use for world state generation.' 
            'choose between ["llama8", "llama70"]'
    ))
    return parser.parse_args()


def main():
    global args
    args = get_args()

    if args.evaluate_all:
        result_files = glob(f'../data/prelim_results/{args.data}/{args.model}/*')
        available_seeds = [int(ele.split('/')[-1]) for ele in result_files]
    else:
        available_seeds = [args.seed]

    vanilla_results_dict, masktom_results_dict, timetom_results_dict = {}, {}, {}
    for seed in available_seeds:
        result_files = glob(f'../data/prelim_results/{args.data}/{args.model}/{seed}/*.json')
        # result_files = [
        #     ele for ele in result_files if args.world_state_model in ele or 'timetom' in ele
        # ]

        timetom_results_dict[seed] = [
            ele for ele in result_files if 'timetom' in ele
        ]
        vanilla_results_dict[seed] = [
            ele for ele in result_files if 'masktom' not in ele and 'timetom' not in ele
        ]
        masktom_results_dict[seed] = [
            ele for ele in result_files if 'masktom' in ele
        ]

    file_io = FileIO()
    eval_config = file_io.load_yaml('./configs/dataset_eval_config.yml')
    eval_prompt = file_io.load_yaml('./prompts/eval_prompt_2_0.yaml')
    eval_method = eval_config[args.data]['eval_method']
    timetom_eval_method = eval_config[args.data]['timetom']

    evaluator = Evaluator(
        eval_method=eval_method, 
        timetom_eval_method=timetom_eval_method, 
        eval_prompt=eval_prompt, 
        data_name=args.data,
        add_label=True,
    )

    timetom_scores = {}
    for seed, timetom_results in timetom_results_dict.items():
        for result_path in timetom_results:

            result_data = file_io.load_json(result_path)
            result_path = result_path.replace('fantom_long', 'fantom-long')
            postfix = '-'.join(result_path.split('/')[-1].strip().rsplit('.', 1)[0].split('_')[1:])
            model_name = result_path.split('/')[-2].strip()

            mc_probing = True if 'mc-probing' in result_path else False

            gt_lst, pred_lst, all_questions = evaluator.evaluate(
                result_data=result_data,
                mc_probing=mc_probing,
                data_name=args.data,
            )

            if not os.path.exists(f'../data/evaluation/labeled_results/{args.model}'):
                os.makedirs(f'../data/evaluation/labeled_results/{args.model}')

            if not os.path.exists(f'../data/evaluation/labeled_results/{args.model}/{args.data}'):
                os.makedirs(f'../data/evaluation/labeled_results/{args.model}/{args.data}')

            if not os.path.exists(f'../data/evaluation/labeled_results/{args.model}/{args.data}/{seed}'):
                os.makedirs(f'../data/evaluation/labeled_results/{args.model}/{args.data}/{seed}')

            cur_fname = result_path.split('/')[-1].split('.')[0]
            file_io.save_json(
                obj=all_questions,
                fpath=f'../data/evaluation/labeled_results/{args.model}/{args.data}/{seed}/{cur_fname}.json',
            )

            acc_num = accuracy_score(gt_lst, pred_lst)
            f1_num = f1_score(gt_lst, pred_lst, average='macro')

            if 'belief-solver' in result_path:
                version = 'timetom+BS'
            else:
                version = 'timetom'

            event_based = '✓' if 'event' in result_path else '✕'
            attr_guided = '✓' if 'curie2' in version else '✕'

            if mc_probing:
                version += '+MC'

            prompt_format = result_path.split('/')[-1].split('_')[1].strip()

            if postfix not in timetom_scores:
                timetom_scores[postfix] = [[
                    'TimeToM',
                    version,
                    'None', # did not use world model
                    event_based,
                    attr_guided,
                    'timetom-text',
                    '✕', # did not use scene graph
                    round(acc_num, 3),
                    round(f1_num, 3),
                ]]
            else:
                timetom_scores[postfix].append([
                    'TimeToM',
                    version,
                    'None', # did not use world model
                    event_based,
                    attr_guided,
                    'timetom-text',
                    '✕', # did not use scene graph
                    round(acc_num, 3),
                    round(f1_num, 3),
                ])

    vanilla_scores = {}
    for seed, vanilla_results in vanilla_results_dict.items():
        for result_path in vanilla_results:

            result_data = file_io.load_json(result_path)
            result_path = result_path.replace('fantom_long', 'fantom-long')

            model_name = result_path.split('/')[-3].strip()

            if 'seed' in result_path:
                postfix = '-'.join(result_path.split('/')[-1].strip().split('.')[0].split('_')[1:-1])
            else:
                postfix = '-'.join(result_path.split('/')[-1].strip().split('.')[0].split('_')[1:])

            gt_lst, pred_lst, all_questions = evaluator.evaluate(
                result_data=result_data,
                data_name=args.data,
            )

            if not os.path.exists(f'../data/evaluation/labeled_results/{args.model}'):
                os.makedirs(f'../data/evaluation/labeled_results/{args.model}')

            if not os.path.exists(f'../data/evaluation/labeled_results/{args.model}/{args.data}'):
                os.makedirs(f'../data/evaluation/labeled_results/{args.model}/{args.data}')

            if not os.path.exists(f'../data/evaluation/labeled_results/{args.model}/{args.data}/{seed}'):
                os.makedirs(f'../data/evaluation/labeled_results/{args.model}/{args.data}/{seed}')

            cur_fname = result_path.split('/')[-1].split('.')[0]
            file_io.save_json(
                obj=all_questions,
                fpath=f'../data/evaluation/labeled_results/{args.model}/{args.data}/{seed}/{cur_fname}.json',
            )

            acc_num = accuracy_score(gt_lst, pred_lst)
            f1_num = f1_score(gt_lst, pred_lst, average='macro')

            version = 'None'

            prompt_format = result_path.split('/')[-1].split('_')[1].rsplit('.', 1)[0].strip()
            event_based = '✓' if 'event' in result_path else '✕'
            attr_guided = '✓' if '-attr' in result_path else '✕'

            postfix = postfix.rsplit('-', 1)[0]
            if postfix not in vanilla_scores:
                vanilla_scores[postfix] = [[
                    'Vanilla',
                    version,
                    'None', # did not use world model
                    event_based,
                    attr_guided,
                    prompt_format,
                    '✕', # did not use scene graph
                    round(acc_num, 3),
                    round(f1_num, 3),
                ]]
            else:
                vanilla_scores[postfix].append([
                    'Vanilla',
                    version,
                    'None', # did not use world model
                    event_based,
                    attr_guided,
                    prompt_format,
                    '✕', # did not use scene graph
                    round(acc_num, 3),
                    round(f1_num, 3),
                ])

    masktom_scores = {}
    for seed, masktom_results in masktom_results_dict.items():
        for result_path in masktom_results:

            result_data = file_io.load_json(result_path)
            result_path = result_path.replace('fantom_long', 'fantom-long')
            postfix = '-'.join(result_path.split('/')[-1].strip().rsplit('.', 1)[0].split('_')[1:])

            gt_lst, pred_lst, all_questions = evaluator.evaluate(
                result_data=result_data,
                data_name=args.data,
            )

            if not os.path.exists(f'../data/evaluation/labeled_results/{args.model}'):
                os.makedirs(f'../data/evaluation/labeled_results/{args.model}')

            if not os.path.exists(f'../data/evaluation/labeled_results/{args.model}/{args.data}'):
                os.makedirs(f'../data/evaluation/labeled_results/{args.model}/{args.data}')

            if not os.path.exists(f'../data/evaluation/labeled_results/{args.model}/{args.data}/{seed}'):
                os.makedirs(f'../data/evaluation/labeled_results/{args.model}/{args.data}/{seed}')

            cur_fname = result_path.split('/')[-1].split('.')[0]
            file_io.save_json(
                obj=all_questions,
                fpath=f'../data/evaluation/labeled_results/{args.model}/{args.data}/{seed}/{cur_fname}.json',
            )

            acc_num = accuracy_score(gt_lst, pred_lst)
            f1_num = f1_score(gt_lst, pred_lst, average='macro')

            framework = 'MaskToM' if '_masktom' in result_path else 'None'
            event_based = '✓' if 'event_based' in result_path else '✕'
            attr_guided = '✓' if '-attr' in result_path else '✕'

            version = 'None'
            scene_graph = '✕'
            if framework != 'None':

                if 'scene_graph' in result_path:
                    version = result_path.split('/')[-1].split('_')[5].strip()
                    scene_graph = '✓'
                else:
                    version = result_path.split('/')[-1].split('_')[5].strip()
                    version = version.rsplit('.', 1)[0]

                wm_model = result_path.split('/')[-1].split('_')[2]

            prompt_format = '|'.join(
                result_path.split('/')[-1].split('_')[1:3]
            ).rsplit('-', 1)[-1].strip()

            scene_graph = '✓' if 'scene_graph' in result_path else '✕'

            if postfix not in masktom_scores:
                masktom_scores[postfix] = [[
                    framework,
                    version,
                    wm_model,
                    event_based,
                    attr_guided,
                    prompt_format,
                    scene_graph,
                    round(acc_num, 3),
                    round(f1_num, 3),
                ]]
            else:
                masktom_scores[postfix].append([
                    framework,
                    version,
                    wm_model,
                    event_based,
                    attr_guided,
                    prompt_format,
                    scene_graph,
                    round(acc_num, 3),
                    round(f1_num, 3),
                ])

    # initialize table
    table = make_table(len(result_data), available_seeds, args.world_state_model)

    pivot_key = f'vanilla-text-{args.world_state_model}'
    pivot_f1_lst = [
        ele[-1] for ele in vanilla_scores[pivot_key]
    ]
    pivot_acc_lst = [
        ele[-2] for ele in vanilla_scores[pivot_key]
    ]
    pivot_f1 = np.mean(pivot_f1_lst)
    pivot_acc = np.mean(pivot_acc_lst)
    pivot_f1_std = np.std(pivot_f1_lst)
    pivot_acc_std = np.std(pivot_acc_lst)

    aggregated_vanilla_scores = {}
    for score_type, score_content in vanilla_scores.items():

        acc_lst = [ele[-2] for ele in score_content]
        f1_lst = [ele[-1] for ele in score_content]

        cur_acc_score = np.mean(acc_lst)
        cur_f1_score = np.mean(f1_lst)
        cur_acc_std = round(np.std(acc_lst), 4)
        cur_f1_std = round(np.std(f1_lst), 4)

        aggregated_vanilla_scores[score_type] = score_content[0][:7] + [
            cur_acc_score,
            cur_f1_score,
            cur_acc_std,
            cur_f1_std,
        ]

    aggregated_vanilla_scores = {
        k: v for k, v in sorted(
            aggregated_vanilla_scores.items(),
            key=lambda item: (item[1][1], item[1][4], item[1][3]),
            reverse=False
    )}

    for key, val in aggregated_vanilla_scores.items():

        delta_acc = val[7] - pivot_acc
        delta_f1 = val[8] - pivot_f1
        delta_acc = round(delta_acc, 3)
        delta_f1 = round(delta_f1, 3)

        if delta_f1 > 0:
            delta_f1 = '+' + str(delta_f1)
        else:
            delta_f1 = str(delta_f1)

        if delta_acc > 0:
            delta_acc = '+' + str(delta_acc)
        else:
            delta_acc = str(delta_acc)

        table.add_row(
            val[0], 
            val[1], 
            val[2], 
            val[3], 
            val[4], 
            val[5], 
            val[6], 
            f'{val[7]:.3f} ± {val[9]:.3f}',
            delta_acc,
        )

    aggregated_timetom_scores = {}
    for score_type, score_content in timetom_scores.items():

        acc_lst = [ele[-2] for ele in score_content]
        f1_lst = [ele[-1] for ele in score_content]

        cur_acc_score = np.mean(acc_lst)
        cur_f1_score = np.mean(f1_lst)
        cur_acc_std = round(np.std(acc_lst), 4)
        cur_f1_std = round(np.std(f1_lst), 4)

        aggregated_timetom_scores[score_type] = score_content[0][:7] + [
            cur_acc_score,
            cur_f1_score,
            cur_acc_std,
            cur_f1_std,
        ]

    table.add_row('', '', '', '', '', '', '')
    for key, val in aggregated_timetom_scores.items():

        if key == pivot_key:
            continue

        delta_acc = val[7] - pivot_acc
        delta_f1 = val[8] - pivot_f1
        delta_f1 = round(delta_f1, 3)
        delta_acc = round(delta_acc, 3)

        if delta_f1 >= 0:
            delta_f1 = '+' + str(delta_f1)
        else:
            delta_f1 = str(delta_f1)

        if delta_acc >= 0:
            delta_acc = '+' + str(delta_acc)
        else:
            delta_acc = str(delta_acc)

        table.add_row(
            val[0], 
            val[1], 
            val[2], 
            val[3], 
            val[4], 
            val[5],
            val[6],
            f'{val[7]:.3f} ± {val[9]:.3f}',
            delta_acc,
        )

    aggregated_masktom_scores = {}
    for score_type, score_content in masktom_scores.items():

        acc_lst = [ele[-2] for ele in score_content]
        f1_lst = [ele[-1] for ele in score_content]

        cur_acc_score = np.mean(acc_lst)
        cur_f1_score = np.mean(f1_lst)
        cur_acc_std = round(np.std(acc_lst), 4)
        cur_f1_std = round(np.std(f1_lst), 4)

        aggregated_masktom_scores[score_type] = score_content[0][:7] + [
            cur_acc_score,
            cur_f1_score,
            cur_acc_std,
            cur_f1_std,
        ]

    aggregated_masktom_scores = {
        k: v for k, v in sorted(
            aggregated_masktom_scores.items(),
            key=lambda item: (item[1][1], item[1][4], item[1][3]),
            reverse=False
    )}

    prev_version = ''
    for key, val in aggregated_masktom_scores.items():

        delta_acc = val[7] - pivot_acc
        delta_f1 = val[8] - pivot_f1
        delta_acc = round(delta_acc, 3)
        delta_f1 = round(delta_f1, 3)

        if delta_f1 > 0:
            delta_f1 = '+' + str(delta_f1)
        else:
            delta_f1 = str(delta_f1)   

        if delta_acc > 0:
            delta_acc = '+' + str(delta_acc)
        else:
            delta_acc = str(delta_acc)

        if val[1] != prev_version:
            table.add_row('', '', '', '', '', '', '')
            prev_version = val[1]

        table.add_row(
            val[0], 
            val[1], 
            val[2], 
            val[3], 
            val[4], 
            val[5],
            val[6],
            f'{val[7]:.3f} ± {val[9]:.3f}',
            delta_acc,
        )

    print('\n\n')
    console = Console(record=True)
    console.print(table)
    print('\n\n')
    console.save_html(f'../data/prelim_results/tables/{args.model}-{args.data}-op.html')

    # Save results to CSV
    save_results_to_csv(
        output_path=f'{args.data}-{args.model}.csv',
        timetom_scores=aggregated_timetom_scores, 
        vanilla_scores=aggregated_vanilla_scores,
        masktom_scores=aggregated_masktom_scores,
    )


if __name__ == '__main__':
    main()
