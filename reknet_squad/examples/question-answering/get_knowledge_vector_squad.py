import configparser
import json
import sys
import numpy as np
import string
from tqdm import tqdm


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or c == "\\" or c == "\"" or ord(c) == 0x202F or\
            c in string.punctuation:
        return True
    return False


if __name__ == "__main__":
    filename = sys.argv[1]
    batch_id = int(sys.argv[2])      # 默认100份，输入-1表示不使用batch
    config = configparser.ConfigParser()
    config.read("paths.cfg")
    res = []
    with open(config["paths"][filename] + "_triple_ids.json", 'r') as reader:
        input_data = json.load(reader)

    if batch_id >= 0:
        output_path = config["paths"][filename] + "_knowledge_vector_%d.json" % batch_id
        batch_data = list(np.array_split(input_data, 100)[batch_id])
    else:
        output_path = config["paths"][filename] + "_knowledge_vector.json"
        batch_data = input_data

    for entry in tqdm(batch_data, desc="bash_id: %d" % batch_id):
        question = entry["question"]
        reference = entry["reference"]
        qas_id = entry["qas_id"]
        knowledge_list = entry["knowledge_list"]

        # get triple
        knowledge_vector = []
        triple_list = []  # 存储未去除glove词汇外的三元组列表，每个元素是"rel||sub||obj||wei"的形式
        operation_list = []
        sub_list = []
        obj_list = []
        wei_list = []
        # 遍历每个三元组，去除重复项
        for knowledge in knowledge_list:
            ls = knowledge.split("||")
            curr_weight = float(ls[3])
            operation_list.append(1)
            for i in range(len(sub_list)):
                if (ls[1] == sub_list[i] and ls[2] == obj_list[i]) or (ls[2] == sub_list[i] and ls[1] == obj_list[i]):
                    if curr_weight > wei_list[i]:
                        operation_list[-1] = 2
                    else:
                        operation_list[-1] = 0
                        break
            sub_list.append(ls[1])
            obj_list.append(ls[2])
            wei_list.append(curr_weight)

        # 去除重复三元组
        for i in range(len(operation_list)):
            if operation_list[i] == 1:
                triple_list.append(knowledge_list[i])
            elif operation_list[i] == 2:
                for triple in triple_list:
                    if triple.find("||" + sub_list[i] + "||" + obj_list[i] + "||") >= 0 or \
                            triple.find("||" + obj_list[i] + "||" + sub_list[i] + "||") >= 0:
                        triple_list.remove(triple)
                        break
                triple_list.append(knowledge_list[i])

        # 将三元组转换为词向量列表[[rel,sub,obj], [rel,sub,obj], ...]
        relation_vectors = {}
        with open(config["paths"]["relation_vector"], "r") as relation_file:
            for line in relation_file.readlines():
                ls = line.strip().split()
                relation_vectors[ls[0]] = ls[1:]
        relation_file.close()

        knowledge_vector_lists = []
        concept_vectors = {}
        for triple in triple_list:
            concept_vectors[triple.split("||")[1]] = []
            concept_vectors[triple.split("||")[2]] = []
        with open(config["paths"]["glove"], "r") as glove_file:
            for line in glove_file.readlines():
                ls = line.strip().split()
                if ls[0] in concept_vectors.keys():
                    concept_vectors[ls[0]] = ls[1:]
        glove_file.close()

        for triple in triple_list:
            ls = triple.split("||")
            if concept_vectors[ls[1]] != [] and concept_vectors[ls[2]] != []:
                knowledge_vector_lists.append([list(map(float, concept_vectors[ls[1]])),
                                               list(map(float, relation_vectors[ls[0]])),
                                               list(map(float, concept_vectors[ls[2]])),
                                               [float(ls[3])] * 100])
        knowledge_vector.append(knowledge_vector_lists)

        res.append({"qas_id": qas_id, "question": question, "reference": reference, "knowledge_vector": knowledge_vector})
    print("Writing to file...")
    with open(output_path, 'w') as writer:
        json.dump(res, writer, indent=2)
