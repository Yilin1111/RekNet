import json
from tqdm import tqdm


count = 0
all_num = 0
know_num = 0
inputfiles = ["train.json", "dev.json", "test.json"]
for inputfile in inputfiles:
    with open(inputfile) as reader:
        input_data = json.load(reader)
    all_num += len(input_data)
    for entry in tqdm(input_data):
        knowledges = entry["knowledge_vector"]
        if knowledges != [[]]:
            count += 1
        for knowledge in knowledges:
            if knowledge:
                know_num += len(knowledge)

print(count)
print("--------------------")
print(all_num)
print("--------------------")
print(know_num)