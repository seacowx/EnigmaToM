for i in 12 42 96 2012 2024
do
      printf "Working on Seed $i\n"
      python generate_states.py --data $1 --model $2 --seed $i
      printf "Working on Seed $i (Event-based)\n"
      python generate_states.py --data $1 --model $2 --seed $i --event_based
done
