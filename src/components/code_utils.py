from ast import literal_eval


def get_location(all_info: list, world_state: dict) -> dict:

    out = {}
    for content in all_info:
        for loc, loc_state in world_state.items():
            if loc.strip() == 'important_private_info':
                continue

            if content[0] in [ele[0] for ele in loc_state]:
                if loc in out.keys():
                    out[loc].append(content)
                else:
                    out[loc] = [content]

    # sort time stamps within each location
    for loc, loc_content in out.items():
        out[loc] = sorted(loc_content, key=lambda x: x[0])

    return out


def code_backoff(char_state: dict) -> str:
    """
    If the code the not executable, backoff to character-centric narrative
    """
    world_state = char_state['world_state']

    fact_narraive_lst = []
    for loc, loc_state in world_state.items():
        temp_narrative = ''
        temp_narrative += f'The following events happened in {loc}:\n'

        for content in loc_state:

            if len(content) > 2:
                if any([ele[0] for ele in content[2].values()]):
                    temp_narrative += content[1] + ' At this moment, '

                    for entity, state_lst in content[2].items():
                        for state in state_lst:
                            try:
                                temp_narrative += f'{entity.capitalize()} is {state[2]}. '
                            except:
                                continue
                else:
                    temp_narrative += content[1]
            else:
                temp_narrative += content[1]

            temp_narrative += '\n\n'

        fact_narraive_lst.append(temp_narrative)

    fact_narrative = '\n'.join(fact_narraive_lst)

    return fact_narrative


def exec_code(ret_code: str, char_state: dict):
    """
    Execute llm returned code and store as local variable
    """
    global_vars = {}

    try:
        exec(ret_code, global_vars)

        retrieved_content = global_vars.get('retrieved_content', [])
        retrieved_entity_states = global_vars.get('retrieved_entity_states', [])
        retrieved_time_stamps = global_vars.get('retrieved_time_stamps', [])

        # sanity check
        valid_output = True
        if not retrieved_content or not retrieved_entity_states:
            valid_output = False

        try:
            assert len(retrieved_content) == len(retrieved_entity_states) == len(retrieved_time_stamps)
        except:
            valid_output = False

        if not valid_output:
            fact_narrative = code_backoff(char_state)

        if isinstance(retrieved_content, str):
            retrieved_content = [retrieved_content]
        if isinstance(retrieved_entity_states, dict):
            retrieved_entity_states = [retrieved_entity_states]

        all_info = [[t, c, e] for t, c, e in zip(retrieved_time_stamps, retrieved_content, retrieved_entity_states)]
        location_dict = get_location(all_info, char_state['world_state'])

        fact_narrative_lst = []
        for location, location_content_lst in location_dict.items():
            temp_narrative = ''
            temp_narrative += f'The following events happened in {location}:\n'
            for location_content in location_content_lst:
                temp_narrative += location_content[1] + ' At this moment, '
                temp_narrative += ' '.join([f'{k.capitalize()} is {v}.' for k, v in location_content[2].items()])
                temp_narrative += '\n\n'

            fact_narrative_lst.append(temp_narrative)

        fact_narrative = '\n'.join(fact_narrative_lst)

    except:
        fact_narrative = code_backoff(char_state)

    return fact_narrative.strip()


def exec_private_code(private_code: str) -> str:
    global_vars = {}
    private_code += '\n\n\nprivate_info = ImportantPrivateInfo()'
    exec(private_code, global_vars)

    retrieved_private_info = global_vars.get('private_info', [])
    # get function in retrieved_private_info that has "time_stemp" in its name
    relevant_methods = [method for method in dir(retrieved_private_info) if 'time_stamp' in method]
    relevant_methods = sorted(relevant_methods, key=lambda x: int(x.split('_')[-1]))

    
    # cannot use list comprehension here
    # https://stackoverflow.com/questions/45194934/eval-fails-in-list-comprehension
    private_tup = []
    for ele in relevant_methods:
        temp_tup = eval(f'retrieved_private_info.{ele}')
        private_tup.append(temp_tup)

    private_narrative = ''
    for content_tup in private_tup:
        private_narrative += content_tup[0] + ' At this moment, '
        private_narrative += ' '.join([f'{k.capitalize()} is {v}.' for k, v in content_tup[1].items()])
        private_narrative += '\n\n'

    return private_narrative.strip()


def extract_question_from_code(code_text: str) -> str:
    return code_text.split('question = ')[-1].replace('"', '').strip()