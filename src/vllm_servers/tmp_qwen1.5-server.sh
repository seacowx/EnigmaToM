vllm serve \
  /scratch_tmp/prj/inf_llmcache/hf_cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --served-model-name Qwen/Qwen2.5-1.5B-Instruct \
  --dtype bfloat16 \
  --api-key seacow \
  --max-model-len 8192 \
  --port 8888 \
  --tensor-parallel-size 1
