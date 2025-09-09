import os
import argparse
from components.utils import FileIO

# import modules from components to use the parsers
from components.get_locations import LocationDetector
from components.get_character_loc import CharacterLocationTracker
from components.get_narrative_locations import NarrativeLocationTracker
from components.get_entity_of_interest import EntityOfInterestExtractor

# narrative location vector generator
from generate_states.generate_narrative_locations import make_narrative_location_vec


def parse_identified_location(
    data_name: str,
    model_name: str,
    identified_location_result: dict,
):
    
    location_lst = []
    for entry in identified_location_result:
        cur_response = entry['response']['body']['choices'][0]['message']['content']
        location_lst.append(cur_response)

    location_detector = LocationDetector(
        model=None,
        async_model=None,
        model_name=model_name,
        prompt_path='',
        dataset_name=data_name, 
        generation_config={},
    )
    narrative_loc_result_parsed = location_detector.parse_narrative_location(
        result_lst=location_lst,
        data_name=data_name
    )

    return narrative_loc_result_parsed


def parse_character_state(
    data_name: str,
    model_name: str,
    character_state_result: list,
    seed: int,
    narrative_len_lst: list,
):
    character_state_lst = []
    for entry in character_state_result:
        cur_response = entry['response']['body']['choices'][0]['message']['content']
        character_state_lst.append(cur_response)

    character_location_tracker = CharacterLocationTracker(
        model=None,
        async_model=None,
        model_name=model_name,
        prompt_path='',
        dataset_name=data_name,
        generation_config={},
        seed=seed,
    )
    
    character_state_parsed = character_location_tracker.parse_character_mask(
        response_lst=character_state_lst,
    )
    character_state_vec = character_location_tracker.make_location_vec(
        location_lst=character_state_parsed,
        narrative_len_lst=narrative_len_lst,
    )

    return character_state_parsed, character_state_vec


def parse_entity_of_interest(
    data_name: str,
    model_name: str,
    entity_of_interest_result: list,
    seed: int,
    char_lst: list,
    add_attr: bool,
):
    eoi_result_lst = []
    for entry in entity_of_interest_result:
        cur_response = entry['response']['body']['choices'][0]['message']['content']
        eoi_result_lst.append(cur_response)

    eoi_extractor = EntityOfInterestExtractor(
        model=None,
        async_model=None,
        model_name=model_name,
        prompt_path='',
        dataset_name=data_name,
        add_attr=add_attr,
    )

    eoi_results = eoi_extractor.parse_eoi(
        result_lst=eoi_result_lst, 
        msg_lst=[],
        char_lst=char_lst,
        add_attr=add_attr,
        do_correct=False,
    )

    return eoi_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        help='Data to process'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='Model used to generate the data'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Seed used to generate the data'
    )
    return parser.parse_args()



def main():

    args = parse_args()
    file_io = FileIO()

    # =================================================================================================================
    # ========================================== process identified location ==========================================
    # =================================================================================================================

    openai_batch_path = (
            f'../data/openai_batch_processed/{args.model}/{args.data}_identify_location_seed{args.seed}.jsonl'
    )

    identified_location_result = file_io.load_jsonl(
        openai_batch_path,
    )

    narrative_loc_result_parsed = parse_identified_location(
        data_name=args.data,
        model_name=args.model,
        identified_location_result=identified_location_result,
    )

    id_lst = [
        entry['custom_id'].split('-')[-1] for entry in identified_location_result
    ]

    location_data = {}
    for i, id in enumerate(id_lst):
        location_data[id] = {
            'detected_locations': narrative_loc_result_parsed[i],
        }

    if not os.path.exists(f'../data/masktom/locations/{args.model}/'):
        os.makedirs(f'../data/masktom/locations/{args.model}/')

    file_io.save_json(
        location_data, 
        (
            f'../data/masktom/locations/{args.model}/' 
            f'{args.data}_samples_seed[{args.seed}].json'
        )
    )

    # =================================================================================================================
    # ============================================ process character state ============================================
    # =================================================================================================================

    openai_batch_path =f'../data/openai_batch_processed/{args.model}/{args.data}_character_state_seed{args.seed}.jsonl'

    if os.path.exists(openai_batch_path):

        character_state_result = file_io.load_jsonl(
            openai_batch_path,
        )

        narrative_len_lst = file_io.load_json(
            f'../data/openai_batch_data/{args.model}/{args.data}_narrative_len_lst_seed{args.seed}.json'
        )

        character_lst = file_io.load_json(
            f'../data/openai_batch_data/{args.model}/{args.data}_character_state_character_lst_seed{args.seed}.json'
        )

        result_parsed, result_vec = parse_character_state(
            data_name=args.data,
            model_name=args.model,
            character_state_result=character_state_result,
            seed=int(args.seed),
            narrative_len_lst=narrative_len_lst,
        )

        j = 0
        perception_data = []
        for k, chars in enumerate(character_lst):

            for char in chars:
                res = result_parsed[j]
                vec = result_vec[j]

                perception_data.append({
                    'id': id_lst[k],
                    'character': char,
                    'response': res,
                    'location_vec': vec, 
                })

                j += 1

        if not os.path.exists(f'../data/masktom/character_state/{args.model}'):
            os.makedirs(f'../data/masktom/character_state/{args.model}')

        file_io.save_json(
            perception_data, 
            f'../data/masktom/character_state/{args.model}/{args.data}_samples_seed[{args.seed}].json'
        )

    # =================================================================================================================
    # ============================================ process Narrative state ============================================
    # =================================================================================================================

    openai_batch_path = (
            f'../data/openai_batch_processed/{args.model}/{args.data}_narrative_states_seed{args.seed}.jsonl'
    )

    if os.path.exists(openai_batch_path):

        narrative_state_result = file_io.load_jsonl(
            openai_batch_path,
        )

        idx_lst = file_io.load_json((
            f'../data/openai_batch_data/' 
            f'{args.model}/{args.data}_narrative_location_id_lst_seed{args.seed}.json'
        ))

        automatic_idx_lst = file_io.load_json((
            f'../data/openai_batch_data/{args.model}/' 
            f'{args.data}_narrative_location_automatic_idx_lst_seed{args.seed}.json'
        ))
        
        narrative_automatic_lst = file_io.load_json((
            f'../data/openai_batch_data/{args.model}/' 
            f'{args.data}_narrative_location_result_automatic_seed{args.seed}.json'
        ))

        narrative_lst = file_io.load_json((
            f'../data/openai_batch_data/{args.model}/' 
            f'{args.data}_narrative_location_narrative_lst_seed{args.seed}.json'
        ))

        available_location_lst = file_io.load_json(( 
            f'../data/openai_batch_data/{args.model}/' 
            f'{args.data}_narrative_location_available_location_lst_seed{args.seed}.json'
        ))

        narrative_location_tracker = NarrativeLocationTracker(
            model=None,
            async_model=None,
            model_name=args.model,
            prompt_path='',
            dataset_name=args.data,
            seed=int(args.seed),
        )

        result_parsed = []
        response_idx = 0
        for narrative_idx in range(len(idx_lst)):
            if narrative_idx in automatic_idx_lst:
                result_parsed.append(
                    narrative_automatic_lst[str(narrative_idx)]
                )
            else:
                result_parsed.append(
                    narrative_location_tracker.parse_narrative_location(
                        response=narrative_state_result[response_idx],
                    ))
                response_idx += 1

        location_data = []
        for j in range(len(id_lst)):
            location_data.append({
                'id': id_lst[j],
                'response': result_parsed[j],
                'available_location': available_location_lst[j],
            })

        location_data = make_narrative_location_vec(
            data=location_data,
            narrative_lst=narrative_lst,
        )

        if not os.path.exists(f'../data/masktom/narrative_loc/{args.model}/'):
            os.makedirs(f'../data/masktom/narrative_loc/{args.model}/')

        file_io.save_json(
            location_data, 
            (
                f'../data/masktom/narrative_loc/{args.model}/'
                f'{args.data}_samples_seed[{args.seed}].json'
        ))

    # =================================================================================================================
    # ========================================== process Entity-of-Interest ===========================================
    # =================================================================================================================

    openai_batch_path = f'../data/openai_batch_processed/{args.model}/{args.data}_eoi_seed{args.seed}.jsonl'

    if os.path.exists(openai_batch_path):

        eoi_result = file_io.load_jsonl(openai_batch_path)

        eoi_char_lst = file_io.load_json(
            f'../data/openai_batch_data/{args.model}/{args.data}_eoi_char_lst_seed{args.seed}.json'
        )

        eoi_question_lst = file_io.load_json(
            f'../data/openai_batch_data/{args.model}/{args.data}_eoi_question_lst_seed{args.seed}.json'
        )

        eoi_id_lst = file_io.load_json(
            f'../data/openai_batch_data/{args.model}/{args.data}_eoi_id_lst_seed{args.seed}.json'
        )

        parsed_eoi_result = parse_entity_of_interest(
            data_name=args.data,
            model_name=args.model,
            entity_of_interest_result=eoi_result,
            seed=int(args.seed),
            char_lst=eoi_char_lst,
            add_attr=False,
        )

        eoi_data = []
        j = 0
        for k, questions in enumerate(eoi_question_lst):

            res = parsed_eoi_result[j]

            eoi_data.append({
                'id': eoi_id_lst[k],
                'questions': questions,
                'response': res,
            })

            j += 1

        if not os.path.exists(f'../data/masktom/eoi/{args.model}'):
            os.makedirs(f'../data/masktom/eoi/{args.model}')

        file_io.save_json(
            eoi_data, 
            f'../data/masktom/eoi/{args.model}/{args.data}_eoi_seed[{args.seed}].json'
        )

        openai_batch_path = f'../data/openai_batch_processed/{args.model}/{args.data}_eoi_attr_seed{args.seed}.jsonl'

        eoi_attr_result = file_io.load_jsonl(openai_batch_path)

        eoi_attr_char_lst = file_io.load_json(
            f'../data/openai_batch_data/{args.model}/{args.data}_eoi_attr_char_lst_seed{args.seed}.json'
        )

        eoi_attr_question_lst = file_io.load_json(
            f'../data/openai_batch_data/{args.model}/{args.data}_eoi_attr_question_lst_seed{args.seed}.json'
        )

        eoi_attr_id_lst = file_io.load_json(
            f'../data/openai_batch_data/{args.model}/{args.data}_eoi_attr_id_lst_seed{args.seed}.json'
        )

        parsed_eoi_attr_result = parse_entity_of_interest(
            data_name=args.data,
            model_name=args.model,
            entity_of_interest_result=eoi_attr_result,
            seed=int(args.seed),
            char_lst=eoi_attr_char_lst,
            add_attr=False,
        )

        eoi_attr_data = []
        j = 0
        for k, questions in enumerate(eoi_attr_question_lst):

            res = parsed_eoi_attr_result[j]

            eoi_attr_data.append({
                'id': eoi_attr_id_lst[k],
                'questions': questions,
                'response': res,
            })

            j += 1

        if not os.path.exists(f'../data/masktom/eoi/{args.model}'):
            os.makedirs(f'../data/masktom/eoi/{args.model}')

        file_io.save_json(
            eoi_attr_data, 
            f'../data/masktom/eoi/{args.model}/{args.data}_attr_eoi_seed[{args.seed}].json'
        )



if __name__ == '__main__':
    main()
