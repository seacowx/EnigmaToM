CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --dtype float16 \
    --api-key seacow \
    --tensor-parallel-size 2 \
    --port 8000 
