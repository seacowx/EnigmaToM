import os,sys
from components.utils import FileIO
from components.get_world_states import WorldStateModels

from vllm.lora.request import LoRARequest
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams


def load_lora_model(world_model_name: str, use_vllm: bool) -> WorldStateModels:

    file_io = FileIO()

    model_info = file_io.load_yaml('./configs/model_info.yml')

    if 'llama70' in world_model_name:
        openpi_entity_lora_path = (
            '/scratch_tmp/prj/charnu/ft_weights/masktom/llama-70b/' 
            'openpi_entity/checkpoint-477'

        )
        openpi_entity_attr_lora_path = (
            '/scratch_tmp/prj/charnu/ft_weights/masktom/llama-70b' 
            '/openpi_entity_attribute'
        )
        quantization = 4

    elif 'llama8' in world_model_name:
        openpi_entity_lora_path = (
            '/scratch_tmp/prj/charnu/ft_weights/masktom/llama-8b/' 
            'openpi_entity/checkpoint-477'
        )
        openpi_entity_attr_lora_path = (
            '/scratch_tmp/prj/charnu/ft_weights/masktom/llama-8b/' 
            'openpi_entity_attribute/checkpoint-747'
        )
        quantization = 16
    else:
        raise ValueError(f'Model {world_model_name} not supported. Please choose from [llama, mixtral]')

    openpi_lora_config = file_io.load_json(
        os.path.join(openpi_entity_lora_path, 'adapter_config.json')
    )         
    openpi_entity_attr_lora_config = file_io.load_json(
        os.path.join(openpi_entity_attr_lora_path, 'adapter_config.json')
    ) 
    base_model_path = openpi_lora_config['base_model_name_or_path']

    world_state_models = WorldStateModels(
        base_model_name = base_model_path,
        openpi_entity_lora_path=openpi_entity_lora_path,
        openpi_entity_attr_lora_path=openpi_entity_attr_lora_path,
        quantization=quantization,
        world_model_name=world_model_name,
        use_vllm=use_vllm,
    )

    return world_state_models
