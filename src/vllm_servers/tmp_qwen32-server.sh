WORLDSIZE=`nvidia-smi --list-gpus | wc -l`
printf "Starting server with $WORLDSIZE GPUs\n"

vllm serve Qwen/Qwen2.5-32B-Instruct \
  --dtype bfloat16 \
  --api-key seacow \
  --port 8888 \
  --served-model-name Qwen/Qwen2.5-32B-Instruct \
  --max-model-len 12000 \
  --gpu_memory_utilization 0.9 \
  --tensor-parallel-size $WORLDSIZE 
