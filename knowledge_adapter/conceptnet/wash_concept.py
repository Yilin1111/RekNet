import json
import sys
from tqdm import tqdm


def process(filename):
    with open(filename, 'r') as reader:
        input_data = json.load(reader)
    res = []
    for entry in tqdm(input_data):
        context = entry["context"]
        question = entry["question"]
        choices_list = entry["choices_list"]
        answer = entry["answer"]
        reference = entry["reference"]
        ori_question = entry["ori_question"]
        entry_id = entry["entry_id"]
        q_concepts = entry["q_concepts"]
        a_concepts = entry["a_concepts"]
        reference = reference.split(" [SEP] ")[0]
        for q_concept in q_concepts:
            if q_concept.replace("_", " ") not in reference:
                q_concepts.remove(q_concept)
        res.append({"context": context, "question": question, "choices_list": choices_list, "answer": answer,
                    "reference": reference, "ori_question": ori_question, "q_concepts": q_concepts,
                    "a_concepts": a_concepts, "entry_id": entry_id})
    print("Writing to file...")
    with open(filename, 'w') as writer:
        json.dump(res, writer)


if __name__ == "__main__":
    input_files = ["dev_concept.json", "test_concept.json", "train_concept.json"]
    for input_file in input_files:
        process(input_file)
