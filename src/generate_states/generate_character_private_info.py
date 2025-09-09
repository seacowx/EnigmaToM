import numpy as np

from components.utils import FileIO
from components.llms import OpenAIInference, OpenAIAsyncInference
from components.get_character_private_info import CharacterPrivateInfoGenerator


async def generate_character_private_info(
    data: dict, 
    augmented_data: dict,
    data_name: str, 
    model: OpenAIInference, 
    async_model: OpenAIAsyncInference,
    model_name: str,
    prompt_path: str, 
    seed: int,
    event_based: bool,
) -> None:

    file_io = FileIO()

    data_character_lst = file_io.load_json(f'../data/masktom/ner/{data_name}_char_name_seed[{seed}].json')

    character_private_info_generator = CharacterPrivateInfoGenerator(
        model=model,
        async_model=async_model,
        model_name=model_name,
        prompt_path=prompt_path,
        dataset_name=data_name,
    )

    narrative_lst = []
    character_lst = []
    id_lst = []
    perception_data = {id: [] for id in data.keys()}
    for (id, entry) in data.items():

        char_lst = data_character_lst[id]

        if event_based:
            narrative = augmented_data[id]['events']
        elif data_name == 'fantom':
            narrative = entry['narrative'].split('\n')
            narrative = [ele for ele in narrative if ele.strip()]
            narrative = [f'{idx+1}: {ele}.' for idx, ele in enumerate(narrative)]
        else:
            narrative = entry['narrative'] + ' '
            narrative = narrative.replace('\n\n', ' ')
            narrative = narrative.replace('\n', ' ')
            narrative = narrative.split('. ')
            narrative = [ele for ele in narrative if ele.strip()]
            narrative = [f'{idx+1}: {ele}.' for idx, ele in enumerate(narrative)]

        narrative_lst.append(narrative)
        character_lst.append(char_lst)
        id_lst.append(id)

    # these outputs should contain every character in the narrative
    result_original, result_parsed, mask_msg_list = await character_private_info_generator.generate(
        narrative_lst, 
        character_lst,
    )

    j = 0
    for k, chars in enumerate(character_lst):

        for l, char in enumerate(chars):

            gen = result_original[j]
            res = result_parsed[j]
            msg = mask_msg_list[j]

            perception_data[id_lst[k]].append({
                'character': char,
                'prompt': msg[-1]['content'],
                'original_generation': gen,
                'response': res,
            })

            j += 1

    postfix = ''
    postfix += '_event' if event_based else ''

    file_io.save_json(
        perception_data, 
        f'../data/masktom/character_private_info/{model_name}/{data_name}{postfix}_samples_seed[{seed}].json'
    )
