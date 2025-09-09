printf "\n\nEvaluating with TimeToM"
python run_exps.py --data $1 --model $2 --world_state_model $3 --run_all --eval_method timetom
python run_exps.py --data $1 --model $2 --world_state_model $3 --run_all --eval_method timetom --use_belief_solver

printf "\n\nEvaluating with DWM"
python run_exps.py --data $1 --model $2 --world_state_model $3 --run_all --eval_method dwm

printf "\n\nEvaluating with PerceptToM"
python run_exps.py --data $1 --model $2 --world_state_model $3 --run_all --eval_method percepttom
