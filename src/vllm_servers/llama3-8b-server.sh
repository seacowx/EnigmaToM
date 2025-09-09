CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /scratch/prj/charnu/seacow_hf_cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --dtype bfloat16 \
  --api-key seacow \
  --port 8888 \
  --served-model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --tensor-parallel-size 1 \
