'''
change to [{context, question, choices_list, answer, entry_id}, ...]
'''

import glob
import tqdm
import json


def get_json(input_dir, out_file):
    lines = []
    files = glob.glob(input_dir + "/*.txt")
    print("Dealing with " + input_dir)
    for file in tqdm.tqdm(files, desc="read files"):
        with open(file, 'r', encoding='utf-8') as fin:
            data_raw = json.load(fin)
            for i in range(len(data_raw["answers"])):
                line = {}
                line["context"] = data_raw["article"]
                line["question"] = data_raw["questions"][i]
                line["choices_list"] = data_raw["options"][i]
                line["answer"] = data_raw["options"][i][ord(data_raw["answers"][i]) - ord('A')]
                line["entry_id"] = file + "%d" % i
                lines.append(line)
    with open(out_file, 'w', encoding='utf-8') as fout:
        json.dump(lines, fout)


input_dirs = ["./train/high", "./train/middle", "./dev/high", "./dev/middle", "./test/high", "./test/middle"]
output_files = ["./train_high.json", "./train_middle.json", "./dev_high.json", "./dev_middle.json", "./test_high.json",
               "./test_middle.json"]
for id in range(len(input_dirs)):
    get_json(input_dirs[id], output_files[id])
