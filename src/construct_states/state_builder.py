from copy import deepcopy
from configs.world_state_config import WorldStateConfig


class StateBuilder:


    def __init__(self, config: WorldStateConfig):
        self.config = config
        self.private_info_dict = {}
        self.skip_key_lst = []

        self.__build_world_state(
            narrative_loc_results=self.config.narrative_loc_info,
            use_scene_graph=self.config.use_scene_graph,
            identified_location_dict=self.config.identified_locations,
        )


    def __make_location_aware_narrative(
        self, 
        narr_loc: dict, 
        indexed_narrative: list,
        identified_location_dict: dict,
    ) -> tuple:

        try:
            narr_loc_vec = narr_loc['narrative_location_vec']
        except:
            # if no location information is provided -> 
            # assume all narratives are in the same location
            cur_location_lst = identified_location_dict[narr_loc['id']]['detected_locations']
            try:
                cur_location = [ele for ele in cur_location_lst if ele != 'location'][0]
            except:
                cur_location = 'location'

            narr_loc_vec = [cur_location for _ in range(len(indexed_narrative)-1)]

        distinct_locations = list(set(narr_loc_vec))

        # indexed_narrative contains either 2 or 3 elements (when entity_state is enabled)
        # 1: index of the narrative
        # 2: the original content of the narrative sentence
        # 3: (optional) fine-grained entity state information
        location_aware_narrative = {loc: [] for loc in distinct_locations}
        narr_loc_vec = narr_loc_vec[:len(indexed_narrative)]
        for idx, loc in enumerate(narr_loc_vec):
            index_n_narrative = indexed_narrative[idx]

            # check if the narrative is in the correct format
            # the first element is index and the second element is the content
            if len(index_n_narrative) < 2:
                continue

            if not isinstance(index_n_narrative[1], str):
                continue

            location_aware_narrative[loc].append(index_n_narrative)

        # correct obvious errors
        for loc in distinct_locations:
            other_locs = [ele for ele in distinct_locations if ele != loc]
            if other_locs:
                for other_loc in other_locs:
                    cur_location_narr = deepcopy(location_aware_narrative[loc])
                    for nar in cur_location_narr:
                        if other_loc.lower() in nar[1].lower():
                            location_aware_narrative[other_loc].append(nar)
                            location_aware_narrative[loc].remove(nar)
        
        return location_aware_narrative, distinct_locations

    
    def __make_location_oblivious_narrative(self, indexed_narrative: list) -> tuple:
        location_aware_narrative = {'location': indexed_narrative}
        return location_aware_narrative, ['location']

    
    def __inject_entity_state(self, indexed_narrative: list, cur_id: str) -> list:
        cur_entity_states = self.config.entity_state_info[cur_id]

        if isinstance(cur_entity_states, list):
            cur_entity_states = cur_entity_states[0][0]
        elif isinstance(cur_entity_states, dict):
            cur_entity_states = cur_entity_states['openpi_states']
        
        cur_entity_states = {
            key: val['entity_state_changes'] for (key, val) in cur_entity_states.items()
        }

        # sanity check
        # assert len(cur_entity_states) == len(indexed_narrative)

        previous_entity_state_dict = {}
        for idx, entity_state_dict in enumerate(cur_entity_states.values()):

            # check if there exists any entity state change @ current time stamp
            valid_entity_state = {
                key: val for (key, val) in entity_state_dict.items() if val
            }

            if valid_entity_state:

                new_valid_entity_state = {}
                for ent, val_lst in valid_entity_state.items():
                    ent = ent.lower()
                    if 'the' in ent:
                        ent = ent.replace('the', '').strip()

                    if self.config.add_attr:
                        if ' of ' in ent:
                            try:
                                attr, ent = [ele.strip() for ele in ent.split(' of ')]
                            except:
                                continue

                        elif "'s" in ent: 
                            attr, ent = [ele.strip() for ele in ent.rsplit("'s", 1)]

                        else:
                            continue

                        # ent = '_'.join(ent.split())
                        # attr = '_'.join(attr.split())

                        new_val_lst = []
                        for ele in val_lst:
                            try:
                                temp_triplet = [ent, attr, ele[-1]]
                                new_val_lst.append(temp_triplet)
                            except:
                                continue
                        new_valid_entity_state[ent] = val_lst
                    else:
                        # ent = '_'.join(ent.split())
                        new_valid_entity_state[ent] = val_lst

                # remove entity states that have been mentioned in the previous time stamp
                unique_valid_entity_state = {}
                for key, val in new_valid_entity_state.items():
                    previous_val = previous_entity_state_dict.get(key, [])
                    if previous_val:
                        val = [ele for ele in val if ele not in previous_val]

                    if val:
                        unique_valid_entity_state[key] = val
                
                previous_entity_state_dict = new_valid_entity_state
                if unique_valid_entity_state:
                    indexed_narrative[idx].append(unique_valid_entity_state)

        return indexed_narrative


    def __build_world_state(
        self, 
        narrative_loc_results: list,
        identified_location_dict: dict,
        use_scene_graph: bool,
    ) -> None:

        self.all_world_states = {}

        # first construct world state from an omniscent perspective
        for narr_loc in narrative_loc_results:

            id = narr_loc['id']
            
            # use extracted events if event-based world state is enabled
            if self.config.event_based:
                indexed_narrative = self.config.augmented_data[id]['events']
            elif self.config.data_name == 'fantom':
                original_narrative = self.config.original_data[id]['narrative'] + ' '
                indexed_narrative = original_narrative.split('\n')
                indexed_narrative = [
                    f'{idx+1}: {ele}.' for idx, ele in enumerate(indexed_narrative)
                ]
            else:
                original_narrative = self.config.original_data[id]['narrative'] + ' '
                original_narrative = original_narrative.replace('\n\n', ' ')
                original_narrative = original_narrative.replace('\n', ' ')
                indexed_narrative = original_narrative.split('. ')
                indexed_narrative = [ele for ele in indexed_narrative if ele.strip()]
                indexed_narrative = [
                    f'{idx+1}: {ele}.' for idx, ele in enumerate(indexed_narrative)
                ]

            if ':' in indexed_narrative[0]:
                indexed_narrative = [ele.split(':', 1) for ele in indexed_narrative]
            elif '.' in indexed_narrative[0]:
                indexed_narrative = [ele.split('.', 1) for ele in indexed_narrative]

            # add fine-grained entity state information
            if self.config.entity_state:

                try:
                    indexed_narrative = self.__inject_entity_state(
                        indexed_narrative=indexed_narrative, 
                        cur_id=id
                    )
                except:
                    self.skip_key_lst.append(id)

            if self.config.implicit_location:
                indexed_narrative, distinct_locations = self.__make_location_oblivious_narrative(
                    indexed_narrative=indexed_narrative
                )

            else:
                indexed_narrative, distinct_locations = self.__make_location_aware_narrative(
                    indexed_narrative=indexed_narrative, 
                    identified_location_dict=identified_location_dict,
                    narr_loc=narr_loc,
                )

            self.all_world_states[id] = {
                'world_state': indexed_narrative,
                'all_locations': distinct_locations,
            }


    def __get_private_info(self) -> None:
        """
        Get characters' private information
        """
        self.private_info_dict = {id: {} for id in self.all_world_states.keys()}
        for id in self.all_world_states.keys():
            private_info = self.config.character_private_info[id]

            private_info_lst = []
            for char_meta_dict in private_info:
                for char, char_dict in char_meta_dict['response'].items():
                    private_info_lst.append({char: list(char_dict.keys())})

            if any(ele.values() for ele in private_info_lst):
                private_info_dict = dict((key,d[key]) for d in private_info_lst for key in d)
                self.private_info_dict[id]['char_private_info'] = private_info_dict

            all_private_idx = [list(ele.values())[0] for ele in private_info_lst]
            all_private_idx = sum(all_private_idx, [])
            all_private_idx.sort()

            self.private_info_dict[id]['all_private_idx'] = all_private_idx


    def __build_text_states(self, char_state_dict: dict, character_lst: list) -> dict:
        """
        Represent the character-centric world state as pure text
        """
        character_world_state_nar = {}
        char_state_dict['location_narrative'] = {}
        for loc, loc_nar_lst in char_state_dict['world_state'].items():

            if not loc_nar_lst:
                continue

            temp_nar = []
            for idx, loc_nar in enumerate(loc_nar_lst):
                if len(loc_nar) == 2:
                    # temp_nar.append(f'## Time Stamp {idx+1}\n- {loc_nar[1]}\n')
                    temp_nar.append(f'{loc_nar[0]}. {loc_nar[1].strip()}')

                    character_world_state_nar[int(loc_nar[0])] = {
                        'content': loc_nar[1],
                        'states': []
                    }

                elif len(loc_nar) == 3:
                    # temp_nar.append(f'## Time Stamp {idx+1}\n- {loc_nar[1]}\n')
                    character_world_state_nar[int(loc_nar[0])] = {
                        'content': loc_nar[1],
                        'states': []
                    }

                    for entity, states in loc_nar[2].items():

                        entity = ' '.join(entity.split('_'))
                        states = [ele for ele in states if ele]

                        if not states:
                            continue

                        # the case where no attribute if provided
                        if not states[0][0]:
                            temp_entity_state_nars = [
                                f'{entity.strip()} becomes {ele[-1].strip()}' for ele in states
                            ]
                            character_world_state_nar[int(loc_nar[0])]['states'].extend(
                                temp_entity_state_nars
                            )

                        # the case where attribute is provided
                        else:
                            try:
                                temp_entity_state_nars = [
                                    f'{ele[1].strip().capitalize()} of {entity} '
                                    f'becomes {ele[-1].strip()}' for ele in states
                                ]
                                temp_entity_state_nars = list(set(temp_entity_state_nars))
                                # temp_nar[-1] = temp_nar[-1].strip()
                                # temp_nar.extend(temp_entity_state_nars)
                                character_world_state_nar[int(loc_nar[0])]['states'].extend(
                                    temp_entity_state_nars
                                )
                            except:
                                continue

        # sort events by their temporal order
        character_world_state_dict = {
            key: val for (key, val) in sorted(character_world_state_nar.items(), key=lambda x: x[0])
        }

        character_world_state_nar = ''
        for val in character_world_state_dict.values():
            character_world_state_nar += f"\n{val['content'].strip()}"
            if cur_state_info_lst := val['states']:
                for state_info in cur_state_info_lst:
                    character_world_state_nar += f" {state_info.strip().capitalize()}."

            character_world_state_nar += '\n'

        character_world_state_nar = character_world_state_nar.strip()

        # # restore time ordering in narrative
        # if character_world_state_nar:
        #     character_world_state_nar = character_world_state_nar.strip().split('\n')
        #     character_world_state_nar = sorted(
        #         character_world_state_nar, key=lambda x: int(x.split(':')[0])
        #     )
        #     character_world_state_nar = '\n'.join([
        #         ele.split('.')[-1].strip() for ele in character_world_state_nar
        #     ])

        if character_lst:
            character_world_state_nar = f'The following events are known to {character_lst[0]}' \
                f'\n\n{character_world_state_nar}'

        char_state_dict['narrative'] = character_world_state_nar.strip()
        return char_state_dict


    def __build_markdown_states(self, char_state_dict: dict, character_lst: list) -> dict:
        """
        Represent the character-centric world state as markdown
        """
        character_world_state_nar = {}
        char_state_dict['location_narrative'] = {}
        for loc, loc_nar_lst in char_state_dict['world_state'].items():

            if not loc_nar_lst:
                continue

            for idx, loc_nar in enumerate(loc_nar_lst):
                if len(loc_nar) == 2:
                    character_world_state_nar[int(loc_nar[0])] = {
                        'content': loc_nar[1], 
                        'states': []
                    }
                elif len(loc_nar) == 3:
                    character_world_state_nar[int(loc_nar[0])] = {
                        'content': loc_nar[1], 
                        'states': []
                    }

                    for entity, states in loc_nar[2].items():

                        entity = ' '.join(entity.split('_'))
                        states = [ele for ele in states if ele]

                        if not states:
                            continue

                        # the case where no attribute if provided
                        if not states[0][0]:
                            temp_entity_state_nars = [
                                f'{entity.strip()} becomes {ele[-1].strip()}' for ele in states
                            ]
                            character_world_state_nar[int(loc_nar[0])]['states'].extend(
                                temp_entity_state_nars
                            )

                        # the case where attribute is provided
                        else:
                            try:
                                temp_entity_state_nars = [
                                    f'{ele[1].strip().capitalize()} of {entity} '
                                    f'becomes {ele[-1].strip()}' for ele in states
                                ]
                                temp_entity_state_nars = list(set(temp_entity_state_nars))
                                # temp_nar[-1] = temp_nar[-1].strip()
                                # temp_nar.extend(temp_entity_state_nars)
                                character_world_state_nar[int(loc_nar[0])]['states'].extend(
                                    temp_entity_state_nars
                                )
                            except:
                                continue

        # sort events by their temporal order
        character_world_state_dict = {
            key: val for (key, val) in sorted(character_world_state_nar.items(), key=lambda x: x[0])
        }

        character_world_state_nar = ''
        for val in character_world_state_dict.values():
            character_world_state_nar += f"- {val['content'].strip()}\n"
            if cur_state_info_lst := val['states']:
                for state_info in cur_state_info_lst:
                    character_world_state_nar += f"\t- {state_info.strip()}\n"

            character_world_state_nar += '\n'

        character_world_state_nar = character_world_state_nar.strip()

        # # restore time ordering in narrative
        # if character_world_state_nar:
        #     character_world_state_nar = character_world_state_nar.strip().split('\n')
        #     character_world_state_nar = sorted(
        #         character_world_state_nar, key=lambda x: int(x.split(':')[0])
        #     )
        #     character_world_state_nar = '\n'.join([
        #         ele.split('.')[-1].strip() for ele in character_world_state_nar
        #     ])

        if character_lst:
            character_world_state_nar = f'- The following events are known to {character_lst[0]}' \
                f'\n\n{character_world_state_nar}'

        char_state_dict['narrative'] = character_world_state_nar.strip()
        return char_state_dict


    # @staticmethod
    # def __add_private_info(
    #     cur_private_info: dict, 
    #     character_world_state: dict, 
    #     cur_world_state: dict
    # ) -> dict:

    #     """
    #     add private information if it is not in the character-centric world state
    #     """ 

    #     character_world_state['important_private_info'] = []
    #     for private_idx in cur_private_info:

    #         private_idx = int(private_idx)

    #         # check if the private index is already in the character-centric world state
    #         for loc, loc_lst in character_world_state.items():
    #             if private_idx in [ele[0] for ele in loc_lst]:
    #                 character_world_state[loc] = [ele for ele in loc_lst if ele[0] != private_idx]

    #         additional_private_info = ['', []]
    #         for loc, loc_lst in cur_world_state.items():
    #             for ele in loc_lst:
    #                 if str(ele[0]) == str(private_idx):
    #                     additional_private_info = [loc, ele]

    #         if additional_private_info[0]:
    #             character_world_state['important_private_info'].append(additional_private_info[1])

    #     character_world_state['important_private_info'].sort(key=lambda x: x[0])

    #     return character_world_state


    # @staticmethod
    # def __remove_private_info(
    #     character_world_state: dict, 
    #     char_private_info: dict, 
    #     character: str
    # ) -> dict:

    #     other_private_idx = [ele for (name, ele) in char_private_info.items() if name != character]
    #     other_private_idx = sum(other_private_idx, [])
        
    #     for loc, loc_lst in character_world_state.items():
    #         if loc == 'important_private_info':
    #             continue
    #         for idx in other_private_idx:
    #             character_world_state[loc] = [ele for ele in loc_lst if ele[0] != idx]

    #     return character_world_state


    # def __adjust_private_info(
    #     self,
    #     character: str, 
    #     character_world_state: dict, 
    #     char_private_info: dict, 
    #     cur_world_state: dict
    # ) -> dict:
    #     """
    #     Add private info for characters that are aware of it
    #     Remove private info form characters that are not aware of it
    #     """

    #     char_lst = list(char_private_info.keys())
    #     if character in char_lst:
    #         cur_private_info = char_private_info[character]

    #         character_world_state = self.__add_private_info(
    #             cur_private_info,
    #             character_world_state,
    #             cur_world_state,
    #         )

    #         character_world_state = self.__remove_private_info(
    #             character_world_state, 
    #             char_private_info, 
    #             character
    #         )

    #     return character_world_state


    def __perspective_taking(
        self,
        character_lst: list,
        entry: dict,
        cur_locations: list,
        character_world_state: dict,
    ):
        for character in character_lst:

            # character's location vector
            location_vec = entry[character]

            for loc in cur_locations:
                if loc in location_vec or ('true' in location_vec and 'false' in location_vec):

                    # if loc does not have any name, assume it to be character mental state
                    if not loc or loc == 'none':
                        loc = 'another_location'

                    if self.config.implicit_location:
                        cur_loc_vec = [
                            str(idx + 1) for (idx, ele) in enumerate(location_vec) if ele.lower() == 'true'
                        ]
                    else:
                        cur_loc_vec = [
                            str(idx + 1) for (idx, ele) in enumerate(location_vec) if ele.lower() == loc.lower()
                        ]

                    if cur_loc_vec:
                        char_loc = character_world_state[loc]
                        character_world_state[loc] = [
                            ele for ele in char_loc if ele[0] in cur_loc_vec
                        ]

            # sort by time stamp within each location
            for loc, loc_lst in character_world_state.items():
                character_world_state[loc] = sorted(loc_lst, key=lambda x: int(x[0]))

        return character_world_state



    def __build_character_states(self, key: str, character_lst: list) -> tuple:
        """
        Construct character-centric world state based on the omniscent world state
        """

        self.character_world_state = {key: []}
        entry = {
            ele['character']: [sub_ele.strip().lower() for sub_ele in ele['location_vec']]
            for ele in self.config.character_loc_info if ele['id'] == key
        }

        if not character_lst:
            return self.all_world_states[key], 'omniscent-perspective'

        cur_locations = self.all_world_states[key]['all_locations'] 
        cur_world_state = self.all_world_states[key]['world_state']
        character_world_state = deepcopy(cur_world_state)

        if self.config.perspective_taking:
            character_world_state = self.__perspective_taking(
                character_lst=character_lst,
                entry=entry,
                cur_locations=cur_locations,
                character_world_state=character_world_state,
            )

        # add and remove private information if the private_info_dict is provided
        # if self.private_info_dict and not self.config.implicit_location:
        #     char_private_info, _ = self.private_info_dict[key].values()

        #     character_world_state = self.__adjust_private_info(
        #         character_lst[0],
        #         character_world_state,
        #         char_private_info,
        #         cur_world_state,
        #     )

        character_perspective = f"{character_lst[-1].capitalize()}'s perspective"

        character_world_state = {
            'character': character_lst,
            'world_state': character_world_state,
        }

        return character_world_state, character_perspective


    def build(self, key: str, character_lst: list):

        # if self.config.private_info:
        #     self.__get_private_info()

        char_state, char_perspective = self.__build_character_states(key, character_lst)

        if self.config.representation == 'markdown':
            char_state = self.__build_markdown_states(char_state, character_lst)

        elif self.config.representation == 'text':
            char_state = self.__build_text_states(char_state, character_lst)

        return char_state, char_perspective
