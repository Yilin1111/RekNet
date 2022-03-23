import configparser
import json
import sys
import numpy as np
import string
from tqdm import tqdm


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or c == "\\" or c == "\"" or ord(c) == 0x202F or \
            c in string.punctuation:
        return True
    return False


if __name__ == "__main__":
    filename = sys.argv[1]
    batch_id = int(sys.argv[2])  # 默认100份，输入-1表示不使用batch
    config = configparser.ConfigParser()
    config.read("paths.cfg")
    res = []
    relations = []
    subjects = []
    objects = []
    weights = []
    with open(config["paths"][filename] + "_concept.json", 'r') as reader:
        input_data = json.load(reader)
    with open(config["paths"]["conceptnet_merge"], 'r') as triple_reader:
        print("Reading ConceptNet file...")
        for line in triple_reader.readlines():
            triple = line.split("\t")
            relations.append(triple[0])
            subjects.append(triple[1])
            objects.append(triple[2])
            weights.append(triple[3])

    if batch_id >= 0:
        output_path = config["paths"][filename] + "_triple_ids_%d.json" % batch_id
        batch_data = list(np.array_split(input_data, 100)[batch_id])
    else:
        output_path = config["paths"][filename] + "_triple_ids.json"
        batch_data = input_data

    for entry in tqdm(batch_data, desc="bash_id: %d" % batch_id):
        context = entry["context"]
        question = entry["question"]
        choices_list = entry["choices_list"]
        answer = entry["answer"]
        reference = entry["reference"]
        ori_question = entry["ori_question"]
        entry_id = entry["entry_id"]
        q_concepts = entry["q_concepts"]
        a_concepts = entry["a_concepts"]

        # save ids of triples for each qa-pair
        sub_qids = []
        obj_qids = []
        knowledge_list = []
        for q_concept in q_concepts:
            # find triples
            for id in range(len(relations)):
                # a_concept is subjects
                if q_concept == subjects[id]:
                    sub_qids.append(id)  # 这个三元组的sub被q包含
                if q_concept == objects[id]:
                    obj_qids.append(id)  # 这个三元组的obj被q包含

        # for each choice's concept
        for a_concept_choice in a_concepts:
            sub_aids = []
            obj_aids = []
            sub_list = []
            obj_list = []
            rel_list = []
            wei_list = []
            # for each concept word of choice's concept
            for a_concept in a_concept_choice:
                # find triples
                for id in range(len(relations)):
                    # a_concept is subjects
                    if a_concept == subjects[id]:
                        sub_aids.append(id)  # 这个三元组的sub被a包含
                    if a_concept == objects[id]:
                        obj_aids.append(id)  # 这个三元组的obj被a包含
            triple_id_list = list((set(sub_qids) & set(obj_aids)) | (set(obj_qids) & set(sub_aids)))
            triple_context_list = []
            for triple_id in triple_id_list:
                # 添加所有三元组、即使有不同的关系
                # is_add: 0：此三元组不应添加    1：此三元组应添加   2：此三元组应替换掉之前的某三元组
                is_add = 1
                curr_weight = float(weights[triple_id].strip())
                for i in range(len(sub_list)):
                    if subjects[triple_id] == sub_list[i] and objects[triple_id] == obj_list[i] and relations[triple_id] == rel_list[i]:
                        if curr_weight > wei_list[i]:
                            wei_list[i] = curr_weight
                            is_add = 2
                        else:
                            is_add = 0
                        break
                if is_add > 0:
                    if is_add == 1:
                        sub_list.append(subjects[triple_id])
                        obj_list.append(objects[triple_id])
                        rel_list.append(relations[triple_id])
                        wei_list.append(curr_weight)
                    if is_add == 2:
                        for triple_context in triple_context_list:
                            if triple_context.find("||" + subjects[triple_id] + "||" + objects[triple_id] + "||") >= 0:
                                triple_context_list.remove(triple_context)
                    triple_context_list.append(relations[triple_id] + "||" + subjects[triple_id] + "||" +
                                               objects[triple_id] + "||" + weights[triple_id].strip())
            knowledge_list.append(triple_context_list)

        res.append({"context": context, "question": question, "choices_list": choices_list, "answer": answer,
                    "reference": reference, "ori_question": ori_question, "knowledge_list": knowledge_list,
                    "entry_id": entry_id})
    print("Writing to file...")
    with open(output_path, 'w') as writer:
        json.dump(res, writer)
