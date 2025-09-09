import json


data = json.load(open('./tomi.json'))

for id, content in data.items():
    containers = content['containers']
    q_dict_lst = content['questions']

    for q_dict in q_dict_lst:
        assert q_dict['answer'] in containers

        other_container = [c for c in containers if c != q_dict['answer']]
        assert len(other_container) == 1

        q_dict['alternative container'] = other_container[0]


with open('./tomi.json', 'w') as f:
    json.dump(data, f, indent=4)


