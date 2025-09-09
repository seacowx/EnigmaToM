import os 
import argparse
from glob import glob
from components.utils import FileIO


def load_correct_letter_lst(cur_path):
    cur_correct_letter_lst_path = cur_path.split('_data.json')[0] + '_correct_letter_lst.json'

    if os.path.exists(cur_correct_letter_lst_path):
        correct_letter_lst = file_io.load_json(cur_correct_letter_lst_path)
        return correct_letter_lst

    return []


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--data', type=str, required=True, help='name of the dataset'
    )
    argparser.add_argument(
        '--seed', type=int, default=2024, help='seed to use for the experiment'
    )
    argparser.add_argument(
        '--model', type=str, default='gpt-4o', help='openai model'
    )
    return argparser.parse_args()


def main():

    global file_io
    args = parse_args()
    file_io = FileIO()

    result_jsonl_file_lst = glob((
            f'../data/openai_batch_exp_processed/{args.model}' 
            f'/{args.data}/{args.seed}/*'
    ))

    for result_jsonl_file in result_jsonl_file_lst:

        eval_type = ''
        save_fname = ''
        if 'vanilla' in result_jsonl_file:
            eval_type = 'vanilla'
            save_fname = 'vanilla-text-llama8_text'
        elif 'cot' in result_jsonl_file:
            eval_type = 'cot'
            save_fname = 'cot-text_llama8_text'

        # NOTE: load eval files
        result_jsonl_file = file_io.load_jsonl(fpath=result_jsonl_file)

        correct_letter_lst = []
        data = {}
        if eval_type == 'vanilla':
            cur_path = (
                f'../data/openai_batch_exp_processed/{args.model}/' 
                f'{args.data}/{args.seed}/{args.data}_vanilla-text.jsonl'
            )
            data = file_io.load_json(
                cur_path
            )
            correct_letter_lst = load_correct_letter_lst(cur_path)

        elif eval_type == 'cot':
            cur_path = (
                f'../data/openai_batch_exp_data/{args.model}/' 
                f'{args.data}/{args.seed}/cot_text_data_data.json'
            )
            data = file_io.load_json(
                cur_path
            )
            correct_letter_lst = load_correct_letter_lst(cur_path)

        cur_result = []
        for result_entry in result_jsonl_file:

            cur_prediction = result_entry['response']['body']['choices'][0]['message']['content']

            cur_result.append(cur_prediction)

        idx = 0
        for key, val in data.items():
            question_lst = val['questions']
            for j, question in enumerate(question_lst):

                data[key]['questions'][j]['predicted'] = cur_result[idx]

                if correct_letter_lst:
                    data[key]['questions'][j]['correct_letter'] = correct_letter_lst[idx]

                idx += 1

        file_io.save_json(
            obj=data,
            fpath=(
                f'../data/prelim_results/{args.data}/{args.model}/{args.seed}/{args.data}_{save_fname}.json'
            ))



if __name__ == '__main__':
    main()
