printf "\n\nEvaluating with Baselines and MaskToM"
printf "\n\nEvaluating with prompt-based location\n"
python run_exps.py --data $1 --model $2 --world_state_model $3 --run_all
# printf "\n\nEvaluating with prompt-based location, event_based\n"
# python run_exps.py --data $1 --model $2 --world_state_model $3 --run_all --event_based
printf "\n\nEvaluating with sceneGraph-based location\n"
python run_exps.py --data $1 --model $2 --world_state_model $3 --run_all --use_scene_graph
# printf "\n\nEvaluating with sceneGraph-based location, event_based\n"
# python run_exps.py --data $1 --model $2 --world_state_model $3 --run_all --event_based --use_scene_graph

