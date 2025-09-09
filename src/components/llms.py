import os
import backoff
from tqdm.asyncio import tqdm
from typing import List, Dict, Optional
from abc import ABCMeta, abstractmethod
import transformers.utils.logging as hf_logging
hf_logging.set_verbosity_error()

import torch
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai import (
    RateLimitError,
    APIError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    BadRequestError,
)
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
os.environ["VLLM_CONFIGURE_LOGGING"] = '0'

MODEL_MAP = {
    'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3-70b': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'llama70-entity-attr': 'llama70-entity-attr',
    'qwen1.5': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen3': 'Qwen/Qwen2.5-3B-Instruct',
    'qwen7': 'Qwen/Qwen2.5-7B-Instruct',
    'qwen14': 'Qwen/Qwen2.5-14B-Instruct',
    'qwen32': 'Qwen/Qwen2.5-32B-Instruct',
    'qwen72': 'Qwen/Qwen2.5-72B-Instruct',
    'gemma9': 'google/gemma-2-9b-it',
    'gemma27': 'google/gemma-2-27b-it',
    'gpt-4o': 'gpt-4o',
}


class LLMBaseModel(metaclass=ABCMeta):

    def init_model(
        self,
        **kwargs,
    ) -> None:
        self.client = OpenAI(
            **kwargs,
        )

    def init_async_model(
        self,
        **kwargs,
    ):
        self.async_client = AsyncOpenAI(
            **kwargs,
        )

    @abstractmethod
    def inference(
        self, 
        model: str,
        message: list,
        temperature: float = 1.0, 
        max_tokens: int = 1024, 
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        do_sample: bool = False,
        return_json: bool = False,
        stream: bool = False,
        json_mode: bool = False,
    ):
        ...


class OpenAIInference(LLMBaseModel):


    def __init__(
        self,
        **kwargs,
    ) -> None:
        self.init_model(
            **kwargs,
        )


    @backoff.on_exception(
        backoff.expo,
        (
            RateLimitError, 
            APIError, 
            APITimeoutError, 
            APIConnectionError, 
            InternalServerError,
            BadRequestError,
        ),
        max_tries=5,
        max_time=70,
    )
    def inference(
        self, 
        model: str,
        message: list,
        temperature: float = 1.0, 
        max_tokens: int = 1024, 
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        json_schema = None,
        do_sample: bool = False,
        stream: bool = False,
    ) -> str | None:

        model_name = MODEL_MAP[model]
        if json_schema:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=message,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                extra_body={
                    'guided_json': json_schema,
                    'guided_decoding_backend': 'outlines',
                },
            )
        else:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=message,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

        try:
            gen_text = response.choices[0].message.content
            return gen_text
        except:
            return ''


class OpenAIAsyncInference(LLMBaseModel):

    def __init__(
        self,
        **kwargs,
    ) -> None:
        self.init_async_model(
            **kwargs,
        )


    async def process_with_semaphore(
        self, 
        semaphore, 
        model: str,
        message: list,
        temperature: float = 1.0, 
        max_tokens: int = 1024, 
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        do_sample: bool = False,
        return_json: bool = False,
        stream: bool = False,
        json_schema = None,
    ):
        async with semaphore:
            return await self.inference(
                model=model,
                message=message,
                temperature=temperature, 
                max_tokens=max_tokens, 
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                do_sample=do_sample,
                return_json=return_json,
                stream=stream,
                json_schema=json_schema,
            )


    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIError, APITimeoutError, APIConnectionError, InternalServerError),
        max_tries=5,
        max_time=70,
    )
    async def inference(
        self, 
        model: str,
        message: list,
        temperature: float = 1.0, 
        max_tokens: int = 1024, 
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        do_sample: bool = False,
        return_json: bool = False,
        stream: bool = False,
        json_schema = None,
    ) -> str | ChatCompletion | None:

        model_name = MODEL_MAP[model]
        if json_schema:
            response = await self.async_client.chat.completions.create(
                model=model_name,
                messages=message,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                extra_body={'guided_json': json_schema},
            )
        else:
            response = await self.async_client.chat.completions.create(
                model=model_name,
                messages=message,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

        if return_json:
            return response
        else:
            return response.choices[0].message.content


class vLLMInference:

    def __init__(
        self, 
        model_name: str,
        quantization: bool,
        enable_lora: bool,
        gpu_memory_utilization: float = 0.75,
        download_dir: str = '',
    ) -> None:

        dtype=torch.bfloat16
        if model_name == 'llama3-8b':
            model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'

        elif model_name == 'mixtral':
            model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

        elif 'llama70' in model_name :
            model_name = 'meta-llama/Llama-3.3-70B-Instruct'

        WORLD_SIZE = torch.cuda.device_count()

        extra_kwargs = {}
        if download_dir:
            extra_kwargs['download_dir'] = download_dir
        if quantization:
            extra_kwargs['quantization'] = "bitsandbytes"
            extra_kwargs['load_format'] = "bitsandbytes"

        self.model = LLM(
            model=model_name,
            disable_log_stats=True,
            dtype=dtype,
            tensor_parallel_size= WORLD_SIZE,
            max_model_len=8192,
            enable_lora=enable_lora,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            **extra_kwargs,
        )

    def vllm_generate(
        self, 
        prompts,
        lora_request: Optional[LoRARequest] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        progress_bar: bool = True,
    ) -> list:

        self.sampling_params = SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            max_tokens=max_tokens,
            skip_special_tokens=True,
        )
        
        tokenizer = self.model.get_tokenizer()
        prompts = tokenizer.apply_chat_template(prompts, tokenize=False)

        outputs = self.model.generate(
            prompts, 
            self.sampling_params,
            lora_request=lora_request,
            use_tqdm=progress_bar,
        )
        out = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            out.append(generated_text)

        return out

