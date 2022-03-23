import json


input_files = ["dev-v1.1.json", "train-v1.1.json"]
input_know_file = "know_triple-v1.1.json"
ans_dict = {}
for input_file in input_files:
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                for qa in paragraph["qas"]:
                    ans_dict[qa["id"]] = qa["answers"][0]["text"]
info_list = []
with open(input_know_file, "r", encoding='utf-8') as reader:
    input_data = json.load(reader)
    for entry in input_data:
        entry["answer"] = ans_dict[entry["qas_id"]]
        info_list.append(entry)

with open("know_triple-v1.11.json", 'w') as writer:
    json.dump(info_list, writer, indent=2)
