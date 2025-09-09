WORLDSIZE=`nvidia-smi --list-gpus | wc -l`
printf "Starting server with $WORLDSIZE GPUs\n"

vllm serve \
  /scratch_tmp/prj/inf_llmcache/hf_cache/models--google--gemma-2-27b-it/snapshots/aaf20e6b9f4c0fcf043f6fb2a2068419086d77b0 \
  --served-model-name google/gemma-2-27b-it \
  --dtype bfloat16 \
  --api-key seacow \
  --max-model-len 8192 \
  --port 8888 \
  --tensor-parallel-size $WORLDSIZE
