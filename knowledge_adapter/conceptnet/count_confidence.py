import json
from tqdm import tqdm


count = 0
know_num = 0
inputfiles = ["train_high.json", "dev_high.json", "test_middle.json", "train_middle.json", "dev_middle.json", "test_high.json"]
for inputfile in inputfiles:
    with open(inputfile) as reader:
        input_data = json.load(reader)
    for entry in tqdm(input_data):
        knowledges = entry["knowledge_vector"]
        for knowledge in knowledges:
            if knowledge:
                know_num += len(knowledge)
                for know in knowledge:
                    if know[-1][0] > 2:
                        count += 1

print(count)
print("--------------------")
print(know_num)