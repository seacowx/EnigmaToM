import os
import json
import numpy as np
from tqdm import tqdm
import networkx as nx
from thefuzz import fuzz

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from components.utils import FileIO
from vllm.lora.request import LoRARequest
from components.get_world_states import WorldStateModels


class SceneGraphGenerator:

    def __init__(self, event_based: bool, world_state_model: WorldStateModels):
        self.file_io = FileIO()
        self.RETURN_KEY_WORDS = [
            'return', 'back', 'go', 'move', 'enter', 're-enter'
        ]
        self.LEAVING_INDICATORS = [
            'leave', 
            'leaving', 
            'absent', 
            'ended', 
            'ending', 
            'depart', 
            'departing', 
            'exit', 
            'exiting',
            'outside',
            'out',
            'away',
            'gone',
        ] 
        self.postfix = '_event' if event_based else ''
        self.world_state_model = world_state_model


    def __normalize_text(self, text: str) -> str:
        text = text.strip()[:-1] if text.strip()[-1] == '.' else text
        text = text.replace('\'s', '').strip()
        return text


    def openpi_entity_attr_prompt(
        self,
        cur_context: str, 
        entity_attr: str, 
    ) -> list:

        entity_attr = entity_attr.replace(':', '').strip()

        dialog = {
            "role": "user",
            "content": (
                f"Event:\n{cur_context}\n\nNow, what happens to the \"{entity_attr}\"? )"
                f"If the {entity_attr} undergoes changes, elicit the new state as bullet points in the " 
                f"following format:\n- {entity_attr.capitalize()} is now [current state].\n\n" 
                f"Otherwise, simply respond with \"None\"."
            )
        }

        return [dialog]


    def __parse_location_info(
        self,
        location_lst: list,
        attr_key_word: str, 
    ) -> list:

        out_location_lst = []
        for predicted_location in location_lst:
            if 'None' in predicted_location:
                out_location_lst.append({})

            else:
                predicted_location = predicted_location.split('<|end_header_id|>')[-1].strip()
                predicted_location_lst = predicted_location.split('\n')
                predicted_location_lst = [ele.lower() for ele in predicted_location_lst]

                # parse response according to finetune template
                for location in predicted_location_lst:

                    cur_entity, cur_location = '', ''
                    if attr_key_word in location:
                        try:
                            cur_entity, cur_location = location.split(attr_key_word)[-1] \
                                .split(' is now ')
                        except:
                            continue
                    elif attr_key_word.capitalize() in location:
                        try:
                            cur_entity, cur_location = location.split(attr_key_word.capitalize())[-1] \
                                .split(' is now ')
                        except:
                            continue

                    cur_location, cur_entity = cur_location.strip(), cur_entity.strip()

                    cur_entity = ' ' + cur_entity + ' '
                    cur_entity = cur_entity.replace(' the ', '').strip()

                    if cur_entity:
                        out_location_lst.append({
                            'entity': cur_entity,
                            'location': cur_location,
                        })

        return out_location_lst


    def __remove_edge(
        self,
        cur_scene_graph: nx.Graph,
        cur_entity: str,
        visited_locations: dict,
    ):
        """
        Removes all edges connected to a specified entity in the scene graph and updates the visited locations.

        Args:
            cur_scene_graph (nx.Graph): The current scene graph from which edges will be removed.
            cur_entity (str): The entity whose connected edges will be removed.
            visited_locations (dict): A dictionary tracking visited locations for each entity.

        Returns:
            tuple: A tuple containing the updated scene graph and the updated visited locations dictionary.
        """
        for edge in cur_scene_graph.edges():
            if cur_entity in edge:
                existing_location = edge[1] if edge[0] == cur_entity else edge[0]
                visited_locations[cur_entity].append(existing_location)
                cur_scene_graph.remove_edge(*edge)

        return cur_scene_graph, visited_locations

    
    def __judge_re_entering(self, cur_sentence: str) -> bool:
        if any([ele in cur_sentence for ele in self.RETURN_KEY_WORDS]):
            return True
        else:
            return False


    def generate_location_info(
        self,
        data_name: str, 
        sampled_data: dict, 
        augmented_data: dict,
        world_model_name: str,
        world_state_model: WorldStateModels,
        seed: int,
        model_name: str,
        use_vllm: bool,
        event_based: bool,
    ):
        if 'llama70' in world_model_name:
            world_model_path = 'llama-70b'
        elif 'llama8' in world_model_name:
            world_model_path = 'llama-8b'
        else:
            raise ValueError('Invalid world model name.')

        # use attribute-guided model to get entity locations
        lora_adapter_path = (
            f'/scratch_tmp/prj/charnu/ft_weights/masktom/{world_model_path}' 
            '/openpi_entity_attribute/checkpoint-747/'
        )

        # Note: Load the character dict and eoi dict
        data_character_dict = self.file_io.load_json(
            f'../data/masktom/ner/{data_name}_char_name_seed[{seed}].json'
        )

        # eoi_lst = self.file_io.load_json(
        #     f'../data/masktom/eoi/{model_name}/{data_name}_eoi_seed[{seed}].json'
        # )
        
        # Note: Organize data for scene graph generation
        narrative_breakdown_dict, narrative_sents_lst = {}, []
        questions_lst = []
        character_lst_lst = []
        id_lst = []
        for idx, (id, entry) in enumerate(sampled_data.items()):
        
            if event_based:
                narrative_sents = augmented_data[id]['events']
                if '.' in narrative_sents[0][:5]:
                    narrative_sents = [n.split('.', 1)[-1].strip() for n in narrative_sents]
                elif ':' in narrative_sents[0][:5]:
                    narrative_sents = [n.split(':', 1)[-1].strip() for n in narrative_sents]
                narrative_sents = [ele for ele in narrative_sents if ele.strip()]

            elif data_name == 'fantom':
                narrative = augmented_data[id]['narrative']
                narrative_sents = narrative.split('\n')
            else:
                narrative = augmented_data[id]['narrative']
                narrative += ' '
                narrative = narrative.replace('\n\n', ' ').replace('\n', ' ')
                narrative_sents = narrative.split('. ')
                narrative_sents = [ele + '.' for ele in narrative_sents if ele.strip()]
                narrative_sents = [ele.replace('..', '.') for ele in narrative_sents]

            # NOTE: cumulative narrative breakdown
            # if data_name == 'fantom':
            #     narrative_breakdown = ['\n'.join(narrative_sents[:i+1]) for i in range(len(narrative_sents))]
            # else:
            #     narrative_breakdown = [' '.join(narrative_sents[:i+1]) for i in range(len(narrative_sents))]

            # NOTE: naive narrative breakdown
            narrative_breakdown = narrative_sents

            character_lst = data_character_dict[id]

            # get set of ToM questions
            questions = entry['questions']
            questions = [ele['question'] for ele in questions]

            # focus on the characters that are mentioned in the questions if not FANToM
            # all characters are important in FANToM for question type [LIST]
            # if data_name != 'fantom':
            #     character_lst = [char for char in character_lst if any([char in q for q in questions])]

            narrative_breakdown_dict[id] = narrative_breakdown
            narrative_sents_lst.append(narrative_sents)
            questions_lst.append(questions)
            character_lst_lst.append(character_lst)
            id_lst.append(id)

        eoi_dict = {}
        for idx, (questions, character_lst) in enumerate(zip(questions_lst, character_lst_lst)):

            if data_name == 'fantom':
                character_attr = ['engagement in the dialogue']
            else:
                character_attr = ['location']

            attr_key_word = character_attr[0] + ' of '

            character_eoi = [
                f"{attr} of {char}" for char in character_lst for attr in character_attr
            ]
            eoi_dict[idx] = list(set(character_eoi))

        msg_lst = []
        msg_idx_to_narrative_idx = {}
        msg_idx_to_eoi = {}
        msg_idx_to_sent_idx = {}
        msg_counter = 0
        for narrative_idx, (id, narrative_sents) in enumerate(tqdm(
                zip(id_lst, narrative_sents_lst), position=0, leave=False
            )):


            narrative_breakdown = narrative_breakdown_dict[id]
            cur_eoi = eoi_dict[narrative_idx]

            for sent_idx, sent in enumerate(narrative_breakdown):
                for eoi in cur_eoi:
                    cur_msg = self.openpi_entity_attr_prompt(
                        sent,
                        eoi,
                    )

                    # NOTE: message only contains the string, not the jsonl format
                    msg_lst.append(cur_msg)
                    msg_idx_to_narrative_idx[msg_counter] = narrative_idx
                    msg_idx_to_eoi[msg_counter] = eoi
                    msg_idx_to_sent_idx[msg_counter] = sent_idx
                    msg_counter += 1

        cache_path = f'../data/scene_graphs/entity_locations/'
        if not os.path.exists(os.path.join(cache_path, world_model_name)):
            os.makedirs(os.path.join(cache_path, world_model_name))

        location_lst = world_state_model.base_model.vllm_generate(
            prompts=msg_lst,
            lora_request=LoRARequest(
                'openpi', 1, lora_adapter_path,
            ),
            temperature=0.,
        )

        location_lst = self.__parse_location_info(
            location_lst=location_lst,
            attr_key_word=attr_key_word,
        )

        out_location_dict = {}
        for msg_idx, location in enumerate(location_lst):
            cur_narrative_idx = msg_idx_to_narrative_idx[msg_idx]
            cur_id = id_lst[cur_narrative_idx]
            cur_eoi = msg_idx_to_eoi[msg_idx]
            cur_sent_idx = msg_idx_to_sent_idx[msg_idx]

            if cur_id not in out_location_dict:
                out_location_dict[cur_id] = {}

            cur_narrative_sents = narrative_sents_lst[cur_narrative_idx]
            cur_sent = cur_narrative_sents[cur_sent_idx]
            
            cur_sent = cur_sent[:-1] if cur_sent[-1] == '.' else cur_sent

            if cur_sent_idx not in out_location_dict[cur_id]:

                out_location_dict[cur_id][cur_sent_idx] = {
                    'sentence': cur_sent,
                    'locations': []
                }

            if location:
                cur_entity, cur_state = location.values()

                if cur_entity.lower() in [ele.lower() for ele in cur_eoi.split()]:
                    cur_state = cur_state[:-1] if cur_state[-1] == '.' else cur_state

                    if cur_entity in cur_sent.lower():
                        out_location_dict[cur_id][cur_sent_idx]['locations'].append(location)

        entity_location_cache_path = os.path.join(
            cache_path, world_model_name, f'{data_name}{self.postfix}_seed[{seed}].json'
        )
        self.file_io.save_json(
            out_location_dict,
            entity_location_cache_path
        )

        return out_location_dict


    def __construct_implicit_scene_graph(self, location_dict: dict, out_location_dict: dict):
        """
        Construct implicit scene graph (binary graph that indicates presence of entity in a location).
        """
        scene_graph_dict = {}
        for id, location_lst in location_dict.items():
            # produce binary vector
            cur_entity_location = out_location_dict[id]

            cur_eoi_lst = list(set(sum(
                [[sub_ele['entity'] for sub_ele in ele['locations']] 
                    for ele in cur_entity_location.values()], [])))

            node_lst = cur_eoi_lst + ['True', 'False']
            cur_scene_graph = nx.Graph()
            cur_scene_graph.add_nodes_from(node_lst)

            # NOTE: assume everyone is absent in the beginning
            for node in cur_eoi_lst:
                cur_scene_graph.add_edge(node, 'False')

            cur_scene_graph_dict = {}
            for timestamp, content_dict in cur_entity_location.items():
                cur_location_dict_lst = content_dict['locations']
                cur_sentence = content_dict['sentence']

                for location_dict in cur_location_dict_lst:
                    raw_entity, raw_location = location_dict.values()

                    cur_entity = self.__normalize_text(raw_entity)
                    cur_location = self.__normalize_text(raw_location)

                    if any([ele in cur_location for ele in self.LEAVING_INDICATORS]):
                        # remove current entity from present
                        if cur_scene_graph.has_edge(cur_entity, 'True'):
                            cur_scene_graph.remove_edge(cur_entity, 'True')
                        # add edge indicating that current entity is absent
                        cur_scene_graph.add_edge(cur_entity, 'False')
                    else:
                        # change indicator from False (absent) to True (present)
                        if cur_scene_graph.has_edge(cur_entity, 'True'):
                            continue
                        elif cur_scene_graph.has_edge(cur_entity, 'False'):
                            cur_scene_graph.remove_edge(cur_entity, 'False')
                            cur_scene_graph.add_edge(cur_entity, 'True')

                cur_scene_graph_edges = nx.node_link_data(cur_scene_graph)
                cur_scene_graph_dict[timestamp] = cur_scene_graph_edges

            scene_graph_dict[id] = cur_scene_graph_dict

        return scene_graph_dict


    def __construct_explicit_scene_graph(self, location_dict: dict, out_location_dict: dict):
        """
        Construct explicit scene graph based on the location information and entity of interest.
        """
        scene_graph_dict = {}
        for id, location_lst in location_dict.items():

            location_lst = location_lst['detected_locations']
            cur_entity_location = out_location_dict[id]

            cur_eoi_lst = list(set(sum(
                [[sub_ele['entity'] for sub_ele in ele['locations']] 
                    for ele in cur_entity_location.values()], [])))

            node_lst = location_lst + cur_eoi_lst

            # initialize current scene graph
            cur_scene_graph = nx.Graph()
            cur_scene_graph.add_nodes_from(node_lst)

            cur_scene_graph_dict = {}
            visited_locations = {
                ent: [] for ent in cur_eoi_lst
            }
            for timestamp, content_dict in cur_entity_location.items():
                cur_location_dict_lst = content_dict['locations']
                cur_sentence = content_dict['sentence']

                for cur_location_dict in cur_location_dict_lst:
                    raw_entity, raw_location = cur_location_dict.values()

                    cur_entity = self.__normalize_text(raw_entity)
                    cur_location = self.__normalize_text(raw_location)

                    if not cur_entity or cur_entity not in visited_locations:
                        continue

                    # check if there is a location change
                    cur_location_tokens = word_tokenize(cur_location)
                    cur_matching_score_lst = [
                        [fuzz.ratio(tok, loc) for tok in cur_location_tokens] for loc in location_lst
                    ]
                    cur_matching_score_lst = [max(ele) for ele in cur_matching_score_lst]

                    try:
                        max_matching_idx = np.argmax(cur_matching_score_lst)
                        max_matching_score = cur_matching_score_lst[max_matching_idx]
                    except:
                        continue

                    if max_matching_score >= 80:
                        matched_location = location_lst[max_matching_idx]
                    elif 'in' in raw_location.split():
                        continue
                    else:
                        matched_location = None

                    # the predicted location is a valid location 
                    if matched_location:

                        # The case when the location has been visited by the entity
                        if matched_location in visited_locations[cur_entity]:

                            # check if the entity is re-entering the location
                            re_entering = self.__judge_re_entering(cur_sentence)
                            if re_entering:
                                # remove other edges and add edge to current location
                                for loc in location_lst:
                                    if cur_scene_graph.has_edge(cur_entity, loc):
                                        cur_scene_graph.remove_edge(cur_entity, loc)
                                cur_scene_graph.add_edge(cur_entity, matched_location)
                                continue

                            # check if the entity is leaving the location
                            cur_location_tokens = word_tokenize(cur_location)
                            if any([ele in cur_location_tokens for ele in self.LEAVING_INDICATORS]):
                                for loc in location_lst:
                                    if cur_scene_graph.has_edge(cur_entity, loc):
                                        cur_scene_graph.remove_edge(cur_entity, loc)
                                continue

                        else:
                            # add edge to connect the entity and the newly visited location
                            for loc in location_lst:
                                if cur_scene_graph.has_edge(cur_entity, loc):
                                    cur_scene_graph.remove_edge(cur_entity, loc)
                                    break
                            cur_scene_graph.add_edge(cur_entity, matched_location)
                            
                            # add new location to the visited list
                            visited_locations[cur_entity].append(matched_location)

                    # else:
                    #     # remove all edges connected to the entity
                    #     # update visited locations: add removed location to the visited list
                    #     cur_scene_graph, visited_locations = self.__remove_edge(
                    #         cur_scene_graph,
                    #         cur_entity,
                    #         visited_locations,
                    #     )

                cur_scene_graph_edges = nx.node_link_data(cur_scene_graph)
                cur_scene_graph_dict[timestamp] = cur_scene_graph_edges
            
            scene_graph_dict[id] = cur_scene_graph_dict

        return scene_graph_dict


    def generate_scene_graph(
        self,
        data_name: str, 
        sampled_data: dict, 
        augmented_data: dict,
        world_model_name: str,
        world_state_model: WorldStateModels,
        seed: int,
        model_name: str,
        use_vllm: bool,
        event_based: bool,
    ):
        location_dict = self.file_io.load_json(
            f'../data/masktom/locations/{model_name}/{data_name}{self.postfix}_samples_seed[{seed}].json'
        )

        cache_path = f'../data/scene_graphs/entity_locations/'
        entity_location_cache_path = os.path.join(
            cache_path, world_model_name, f'{data_name}{self.postfix}_seed[{seed}].json'
        )
        if not os.path.exists(entity_location_cache_path):
            out_location_dict = self.generate_location_info(
                data_name=data_name, 
                sampled_data=sampled_data, 
                augmented_data=augmented_data,
                world_model_name=world_model_name,
                world_state_model=world_state_model,
                seed=seed,
                model_name=model_name,
                use_vllm=use_vllm,
                event_based=event_based
            )
        else:
            out_location_dict = self.file_io.load_json(entity_location_cache_path)

        # use presence-based implicit spatial information 
        if data_name == 'fantom':
            scene_graph_dict = self.__construct_implicit_scene_graph(
                location_dict=location_dict,
                out_location_dict=out_location_dict,
            )
        # use location-based explicit spatial information 
        else:
            scene_graph_dict = self.__construct_explicit_scene_graph(
                location_dict=location_dict,
                out_location_dict=out_location_dict,
            )

        if not os.path.exists(f'../data/scene_graphs/character_graph/{world_model_name}'):
            os.makedirs(f'../data/scene_graphs/character_graph/{world_model_name}')

        self.file_io.save_json(
            scene_graph_dict,
            f'../data/scene_graphs/character_graph/{world_model_name}/' 
            f'{data_name}{self.postfix}_seed[{seed}].json'
        )

    
    def vectorize_graph(self, model_name: str, world_model_name: str, data_name: str, seed: int):
        """
        Converts scene graph data into location vectors for each character.

        This function processes scene graph data to create a time-series of locations for each character.
        It reads character and scene graph data from JSON files and outputs location vectors showing
        where each character is at different timestamps.

        Args:
            model_name (str): Name of the model (currently unused)
            world_model_name (str): Name of the world model, used for file paths
            data_name (str): Name of the dataset
            seed (int): Random seed used for data generation

        Returns:
            None: Saves the output to a JSON file containing a list of dictionaries with:
                - id: Scene identifier
                - character: Character name
                - location_vec: List of locations for the character across timestamps

        Files:
            Reads from:
                - ../data/masktom/ner/{data_name}_char_name_seed[{seed}].json
                - ../data/scene_graphs/character_graph/{world_model_name}/{data_name}{postfix}_seed[{seed}].json
            Writes to:
                - ../data/scene_graphs/character_locations/{world_model_name}/{data_name}{postfix}_seed[{seed}].json
        """

        data_character_dict = self.file_io.load_json(
            f'../data/masktom/ner/{data_name}_char_name_seed[{seed}].json'
        )

        scene_graph = self.file_io.load_json(
            f'../data/scene_graphs/character_graph/{world_model_name}/'
            f'{data_name}{self.postfix}_seed[{seed}].json'
        )

        out_location_vector_lst = []
        for id, scene_graph_dict in scene_graph.items():

            cur_characters = data_character_dict[id]

            for char_name in cur_characters:
                char_location_vec = []
                for timestamp, scene_graph in scene_graph_dict.items():
                    edge_lst = scene_graph['links']

                    cur_loc = 'none'
                    for ele_dict in edge_lst:
                        if ele_dict['target'].lower() == char_name.lower():
                            cur_loc = ele_dict['source']
                            break
                        elif ele_dict['source'].lower() == char_name.lower():
                            cur_loc = ele_dict['target']
                            break

                    char_location_vec.append(cur_loc)

                out_location_vector_lst.append({
                    'id': id,
                    'character': char_name,
                    'location_vec': char_location_vec,
                })

        if not os.path.exists(
            f'../data/scene_graphs/character_locations/{world_model_name}'
        ):
            os.makedirs(
                f'../data/scene_graphs/character_locations/{world_model_name}'
            )
        self.file_io.save_json(
            out_location_vector_lst,
            f'../data/scene_graphs/character_locations/{world_model_name}/'
            f'{data_name}{self.postfix}_seed[{seed}].json'
        )

def generate_scene_graph(
    data_name: str, 
    sampled_data: dict, 
    augmented_data: dict,
    world_model_name: str,
    world_state_model: WorldStateModels,
    seed: int,
    model_name: str,
    use_vllm: bool,
    event_based: bool,
):
    
    scene_graph_generator = SceneGraphGenerator(
        event_based=event_based,
        world_state_model=world_state_model,
    )
    scene_graph_generator.generate_scene_graph(
        data_name=data_name, 
        sampled_data=sampled_data, 
        augmented_data=augmented_data,
        world_model_name=world_model_name,
        world_state_model=world_state_model,
        seed=seed,
        model_name=model_name,
        use_vllm=use_vllm,
        event_based=event_based,
    )

    scene_graph_generator.vectorize_graph(
        model_name=model_name, 
        world_model_name=world_model_name,
        data_name=data_name, 
        seed=seed,
    )
