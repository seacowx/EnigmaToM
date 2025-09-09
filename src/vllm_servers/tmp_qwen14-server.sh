vllm serve Qwen/Qwen2.5-14B-Instruct \
  --dtype bfloat16 \
  --api-key seacow \
  --max-model-len 8192 \
  --port 8888 \
  --served-model-name Qwen/Qwen2.5-14B-Instruct \
  --tensor-parallel-size 1 
