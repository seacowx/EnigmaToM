# reduce tom question to normal zeroth-order question
python reduce_tom_questions.py --data_path ../data/tom_datasets/tomi/tomi.json

# ==============================================
# CODE FOR RUNNING EXPERIMENTS
# ==============================================
# llama-7b, NO MaskToM, Vanilla prompt
python baseline_exp.py --data_path ../data/tom_datasets/tomi/tomi_reduced_q.json --model llama-7b --prompt_type vanilla-text

# llama-7b, MaskToM-ada, Python prompt
python baseline_exp.py --data_path ../data/tom_datasets/tomi/tomi_reduced_q.json --model llama-7b --masktom --masktom_version ada -r python

# llama-7b, NO MaskToM, Python prompt
python baseline_exp.py --data_path ../data/tom_datasets/tomi/tomi_reduced_q.json --model llama-7b -r python

# llama-7b, NO MaskToM, SimToM prompt
python baseline_exp.py --data_path ../data/tom_datasets/tomi/tomi_reduced_q.json --model llama-7b --prompt_type simtom-text

# llama-7b, NO MaskToM, CoT prompt
python baseline_exp.py --data_path ../data/tom_datasets/tomi/tomi_reduced_q.json --model llama-7b --prompt_type cot-text