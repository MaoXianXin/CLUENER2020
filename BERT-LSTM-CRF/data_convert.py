import json

with open('train1.json', 'r') as f:
    d = json.load(f)

with open('test_data.json', 'w', encoding='utf-8') as json_file:
    for task in d:
        dict_one = {}
        tech_dict = {}
        dict_one['text'] = task['text']

        try:
            for label in task['label']:
                if label['labels'][0] not in tech_dict.keys():
                    tech_dict[label['labels'][0]] = {label['text']: [[label['start'], label['end'] - 1]]}
                elif label['labels'][0] in tech_dict.keys():
                    if label['text'] not in tech_dict[label['labels'][0]].keys():
                        tech_dict[label['labels'][0]][label['text']] = [[label['start'], label['end'] - 1]]
                    else:
                        tech_dict[label['labels'][0]][label['text']].append([label['start'], label['end'] - 1])
            dict_one['label'] = tech_dict

            json_str = json.dumps(dict_one)
            json.dump(dict_one, json_file, ensure_ascii=False)
            json_file.write('\n')
        except KeyError:
            pass
