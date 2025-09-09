printf "\n\nGenerating States for $1 with $2\n"
python generate_states.py --data $1 --model $2 
python generate_states.py --data $1 --model $2 --event_based
