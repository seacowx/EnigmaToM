from components.utils import FileIO
from construct_states.state_builder import StateBuilder
from configs.world_state_config import WorldStateConfig

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def init_state_builder(
    model: str, 
    data_name: str, 
    representation: str, 
    version: str,
    world_state_model: str,
    seed: int,
    event_based: bool,
    use_scene_graph: bool,
):

    '''
    init_state_builder initialize the character state builder

    Args:
        representation (str): type of representation to use, choose from [TOML, Python, Markdown]
        version (str): version of MaskToM to use, choose from [ada, babbage, curie1, curie2, davinci1, davinci2]
        ada: narrative location + character location
        babbage: narrative location + character location + character private info
        curie1: narrative location + character location + character private info + entity state
        curie2: narrative location + character location + character private info + entity state + add_attr
        WARNING: davinci models are not supported yet
    '''

    file_io = FileIO()

    valid_versions = ['ada', 'babbage1', 'babbage2', 'curie1', 'curie2']

    assert version in valid_versions, f'Version {version} not supported. Please choose from {valid_versions}'

    perspective_taking = False
    entity_state = False
    add_attr = False

    if version == 'ada':
        perspective_taking = True 
        entity_state = False
        add_attr = False
    elif version == 'babbage1':
        perspective_taking = False
        entity_state = True
        add_attr = False
    elif version == 'babbage2':
        perspective_taking = False
        entity_state = True
        add_attr = True
    elif version == 'curie1':
        perspective_taking = True
        entity_state = True
        add_attr = False
    elif version == 'curie2':
        perspective_taking = True
        entity_state = True
        add_attr = True

    postfix = ''
    postfix += '_event' if event_based else ''

    identified_locations = file_io.load_json(
        f'../data/masktom/locations/{model}/{data_name}_samples_seed[{seed}].json'
    )

    if use_scene_graph:
        character_loc_results = file_io.load_json((
            f'../data/scene_graphs/character_locations/'
            f'{world_state_model}/{data_name}{postfix}_seed[{seed}].json'
        ))
    else:
        character_loc_results = file_io.load_json((
            f'../data/masktom/character_state/{model}/'
            f'{data_name}{postfix}_samples_seed[{seed}].json'
        ))

    narrative_loc_results = file_io.load_json((
        f'../data/masktom/narrative_loc/{model}/'
        f'{data_name}{postfix}_samples_seed[{seed}].json'
    ))

    postfix = ''
    postfix += '_event' if event_based else ''
    postfix += '_attr' if add_attr else ''

    entity_state_info = file_io.load_json(
        f'../data/world_state/{model}/{data_name}/{world_state_model}{postfix}_seed[{seed}].json'
    )

    original_data = file_io.load_json(
        f'../data/tom_datasets/{data_name.lower()}/{data_name.lower()}_reduced_q_seed[{seed}].json'
    )

    augmented_data = {}
    if event_based:
        augmented_data = file_io.load_json(
            f'../data/augmented_data/{model}/{data_name.lower()}_augmented_seed[{seed}].json'
        )

    implicit_location = False
    if data_name in ['fantom'] and use_scene_graph:
        implicit_location = True

    config = WorldStateConfig(
        representation=representation,
        entity_state=entity_state,
        perspective_taking=perspective_taking,
        narrative_loc_info=narrative_loc_results,
        character_loc_info=character_loc_results,
        identified_locations=identified_locations,
        entity_state_info=entity_state_info,
        add_attr = add_attr,
        data_name=data_name.lower(),
        original_data=original_data,
        augmented_data=augmented_data,
        event_based=event_based,
        use_scene_graph=use_scene_graph,
        implicit_location=implicit_location,
    )

    return StateBuilder(config)
