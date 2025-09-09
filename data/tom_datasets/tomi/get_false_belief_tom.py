import json


with open('./tomi.json') as f:
    full_tomi = json.load(f)

out_data = {}
question_types = set()
for key, val in full_tomi.items():
    question_types.update([ele['question_type'] for ele in val['questions']])
    new_questions = []
    for q_dict in val['questions']:
        if q_dict['question_type'] not in ['memory', 'reality']:
            new_questions.append(q_dict)

    if new_questions:
        val['questions'] = new_questions
        out_data[key] = val

with open('./tomi_fb.json', 'w') as f:
    json.dump(out_data, f, indent=4)