vllm serve \
  /scratch_tmp/prj/inf_llmcache/hf_cache/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819 \
  --served-model-name google/gemma-2-9b-it \
  --dtype bfloat16 \
  --api-key seacow \
  --max-model-len 8192 \
  --port 8888 \
  --tensor-parallel-size 1
