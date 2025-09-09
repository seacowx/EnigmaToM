python generate_states.py --data $1 --model llama3-8b --world_state --world_state_model $2 --use_vllm
python generate_states.py --data $1 --model llama3-8b --world_state --world_state_model $2 --use_vllm --attr_guided
python generate_states.py --data $1 --model llama3-8b --world_state --world_state_model $2 --use_vllm --event_based
python generate_states.py --data $1 --model llama3-8b --world_state --world_state_model $2 --use_vllm --event_based --attr_guided
