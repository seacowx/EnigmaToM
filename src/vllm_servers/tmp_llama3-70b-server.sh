WORLDSIZE=`nvidia-smi --list-gpus | wc -l`
printf "Starting server with $WORLDSIZE GPUs\n"

vllm serve \
  /scratch_tmp/prj/inf_llmcache/hf_cache/models--unsloth--Llama-3.3-70B-Instruct-bnb-4bit/snapshots/74be54198eaf4f3c7fba1f4e9fa63725a810c7eb \
  --served-model-name meta-llama/Meta-Llama-3-70B-Instruct \
  --load-format bitsandbytes \
  --quantization bitsandbytes \
  --api-key seacow \
  --max-model-len 8192 \
  --port 8888 \
  --enforce-eager \
  --gpu_memory_utilization 0.8 \
  --pipeline-parallel-size $WORLDSIZE 
  # --tensor-parallel-size $WORLDSIZE 
