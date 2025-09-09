vllm serve \
  /scratch_tmp/prj/inf_llmcache/hf_cache/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --served-model-name Qwen/Qwen2.5-3B-Instruct \
  --dtype bfloat16 \
  --api-key seacow \
  --max-model-len 8192 \
  --port 8888 \
  --tensor-parallel-size 1
