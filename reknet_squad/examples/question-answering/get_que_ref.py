import json
from tqdm import tqdm


input_file = "dev-v1.1.json"
context_dict = {}
with open(input_file, "r", encoding='utf-8') as reader:
    input_data = json.load(reader)["data"]
    for entry in tqdm(input_data):
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                context_dict[qa["id"]] = context

with open("context_dev-v1.1.json", 'w') as writer:
    json.dump(context_dict, writer, indent=2)
