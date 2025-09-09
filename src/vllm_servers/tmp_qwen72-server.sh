WORLDSIZE=`nvidia-smi --list-gpus | wc -l`
printf "Starting server with $WORLDSIZE GPUs\n"

vllm serve \
  /scratch_tmp/prj/charnu/seacow_hf_cache/models--unsloth--Qwen2.5-72B-Instruct-bnb-4bit/snapshots/95cde7b0316fd420d6fb7496c41f56fb9a1711d3 \
  --load-format bitsandbytes \
  --quantization bitsandbytes \
  --api-key seacow \
  --port 8888 \
  --served-model-name Qwen/Qwen2.5-72B-Instruct \
  --max-model-len 12000 \
  --gpu_memory_utilization 0.9 \
  --enforce-eager \
  --pipeline-parallel-size $WORLDSIZE 
  # --max-model-len 8192 \
