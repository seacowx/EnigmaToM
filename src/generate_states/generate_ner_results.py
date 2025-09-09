from components.utils import FileIO
from components.get_ner import NERModule


def generate_ner_results(data: dict, data_name: str, seed: int) -> None:
    file_io = FileIO()

    #NOTE: extracting character names from narrative and produce route vector
    ner_module = NERModule()

    sampled_data_sents = {key: val['narrative'].replace('\n\n', ' ') for key, val in data.items()}
    if data_name == 'fantom':
        sampled_data_sents = {key: val['narrative'].split('\n') for key, val in data.items()}
    else:
        sampled_data_sents = {key: val['narrative'].split('. ') for key, val in data.items()}

    route_vec, data_character_dict = ner_module.ner_inference(
        narrative_sents=sampled_data_sents,
        data_name=data_name
    )

    file_io.save_json(
        route_vec, f'../data/masktom/ner/{data_name}_route_vec_seed[{seed}].json'
    )
    file_io.save_json(
        data_character_dict, f'../data/masktom/ner/{data_name}_char_name_seed[{seed}].json'
    )
