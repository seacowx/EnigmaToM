printf "\n\nGenerating Entity States for $1 with $2 and NKG is $3, Entity-Guided\n"
python generate_states.py \
  --data $1 \
  --model $2 \
  --world_state \
  --world_state_model $3 \
  --use_vllm

printf "\n\nGenerating Entity States for $1 with $2 and NKG is $3, Entity-Attr-Guided\n"
python generate_states.py \
  --data $1 \
  --model $2 \
  --world_state \
  --world_state_model $3 \
  --attr_guided \
  --use_vllm

# printf "\n\nGenerating Entity States for $1 with $2 and NKG is $3, Entity-Guided, Event-Based\n"
# python generate_states.py \
#   --data $1 \
#   --model $2 \
#   --world_state \
#   --world_state_model $3 \
#   --event_based \
#   --use_vllm 
#
# printf "\n\nGenerating Entity States for $1 with $2 and NKG is $3, Entity-Guided, Event-Attr-Based\n"
# python generate_states.py \
#   --data $1 \
#   --model $2 \
#   --world_state \
#   --world_state_model $3 \
#   --event_based \
#   --attr_guided \
#   --use_vllm 
#
# # # =======================================================================================================
# # # =========================================Scene Graph===================================================
# # # =======================================================================================================
# #
# printf "\n\nGenerating Scene Graph for $1 with $2 and NKG is $3\n"
# python generate_states.py \
#   --data $1 \
#   --model $2 \
#   --scene_graph \
#   --world_state_model $3 \
#   --use_vllm 
#
# printf "\n\nGenerating Scene Graph for $1 with $2 and NKG is $3, Event-Based\n"
# python generate_states.py \
#   --data $1 \
#   --model $2 \
#   --scene_graph \
#   --world_state_model $3 \
#   --event_based \
#   --use_vllm 
