import json
import random


def generate_unique_id():
    used_numbers = set()
    
    def generate_single():
        # Generate number between 1000000000 and 9999999999
        return random.randint(1000000000, 9999999999)
    
    while True:
        new_number = generate_single()
        if new_number not in used_numbers:
            used_numbers.add(new_number)
            return new_number


def main():

    data = json.load(open('./fantom_full.json'))
    random.seed(2024)

    visited_narratives = set()
    narrative_to_id = {}
    out_data = {}
    question_type_counts = {}
    fact_dict = {}
    for entry in data:
        cur_narrative = entry['narrative']
        cur_question = {k: v for k, v in entry.items() if k != 'narrative'}

        if cur_narrative not in visited_narratives:
            visited_narratives.add(cur_narrative)
            cur_id = generate_unique_id()
            out_data[cur_id] = {
                'narrative': cur_narrative,
                'questions': []
            }
            narrative_to_id[cur_narrative] = cur_id
            question_type_counts[cur_id] = {}
        else:
            cur_id = narrative_to_id[cur_narrative]

        cur_question_type_count = question_type_counts[cur_id]

        cur_question_type = cur_question['question_type']
        if cur_question_type == 'fact':
            fact_dict[cur_id] = {
                'question': cur_question['question'],
                'answer': cur_question['correct_answer'],
            }

    for entry in data:
        cur_narrative = entry['narrative']
        cur_question = {k: v for k, v in entry.items() if k != 'narrative'}

        if cur_narrative not in visited_narratives:
            visited_narratives.add(cur_narrative)
            cur_id = generate_unique_id()
            out_data[cur_id] = {
                'narrative': cur_narrative,
                'questions': []
            }
            narrative_to_id[cur_narrative] = cur_id
            question_type_counts[cur_id] = {}
        else:
            cur_id = narrative_to_id[cur_narrative]

        cur_question_type_count = question_type_counts[cur_id]

        cur_question_type = cur_question['question_type']

        if cur_question_type == 'fact':
            continue

        if 'tom_type' in cur_question:
            cur_question_type += ' | ' + cur_question['tom_type']

        if 'info_accessibility' in cur_question_type:
            fact_info = fact_dict[cur_id]['answer']
            question = cur_question['question']
            original_punc = question[-1]

            question = question.split('this information')[0]
            question += f'the aforementioned information{original_punc}'
            question = f'Information: {fact_info}\n{question}'

            # cur_question['question'] = f"Information: {fact_info}\n{cur_question['question']}"
            cur_question['question'] = question

        elif 'answerability' in cur_question_type:
            fact_info = fact_dict[cur_id]['question']
            question = cur_question['question']
            original_punc = question[-1]

            question = question.split('this question')[0]
            question += f'the aforementioned question{original_punc}'
            question = f'Question: {fact_info}\n{question}'

            cur_question['question'] = question

        if cur_question_type not in cur_question_type_count:
            cur_question_type_count[cur_question_type] = 0

        cur_question_type_count[cur_question_type] += 1

        if cur_question_type_count[cur_question_type] <= 2:
            out_data[cur_id]['questions'].append(cur_question)

    with open('./fantom_long.json', 'w') as f:
        json.dump(out_data, f, indent=4)


if __name__ == '__main__':
    main()
