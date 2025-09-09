CUDA_VISIBLE_DEVICES=0 python generate_states.py --data $1 --model llama3-8b --scene_graph --world_state_model $2 --use_vllm
CUDA_VISIBLE_DEVICES=0 python generate_states.py --data $1 --model llama3-8b --scene_graph --world_state_model $2 --use_vllm --event_based
