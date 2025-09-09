from pydantic import BaseModel


class WorldStateConfig(BaseModel):
    representation: str
    # private_info: bool
    entity_state: bool
    perspective_taking: bool
    identified_locations: dict
    # character_private_info: dict
    narrative_loc_info: list
    character_loc_info: list
    entity_state_info: dict
    add_attr: bool
    data_name: str
    event_based: bool
    use_scene_graph: bool
    implicit_location: bool
    original_data: dict
    augmented_data: dict = {}
