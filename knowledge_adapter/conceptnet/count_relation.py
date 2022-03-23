import json
from tqdm import tqdm


relation_dic = {'is': 0, 'have': 0, 'can': 0, 'leader': 0, 'capital': 0, 'field': 0, 'situation': 0, 'language': 0,
                'related': 0, 'product': 0, 'influence': 0, 'causes': 0, 'occupation': 0, 'unnecessary': 0, 'locate': 0,
                'desires': 0, 'isa': 0, 'disable': 0, 'entails': 0, 'antonym': 0, 'error': 0}
know_num = 0
inputfiles = ["train_high.json", "dev_high.json", "test_middle.json", "train_middle.json", "dev_middle.json", "test_high.json"]
index1 = []
index2 = []
for inputfile in inputfiles:
    with open(inputfile) as reader:
        input_data = json.load(reader)
    for entry in tqdm(input_data):
        knowledges = entry["knowledge_vector"]
        for knowledge in knowledges:
            if knowledge:
                for know in knowledge:
                    tmp = know[1][0]
                    tmp1 = know[0][0]
                    tmp2 = know[2][0]
                    if tmp == -0.25922:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['related'] += 1
                    elif tmp == -0.27155:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['causes'] += 1
                    elif tmp == 0.15711:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['have'] += 1
                    elif tmp == -0.71766:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['can'] += 1
                    elif tmp == -0.49836:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['leader'] += 1
                    elif tmp == -0.3771:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['capital'] += 1
                    elif tmp == -0.10788:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['field'] += 1
                    elif tmp == -0.39036:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['situation'] += 1
                    elif tmp == 0.18519:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['language'] += 1
                    elif tmp == -0.54264:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['is'] += 1
                    elif tmp == 0.12804:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['product'] += 1
                    elif tmp == 0.31131:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['influence'] += 1
                    elif tmp == -0.20053:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['occupation'] += 1
                    elif tmp == -0.82866:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['unnecessary'] += 1
                    elif tmp == -0.49679:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['locate'] += 1
                    elif tmp == 0.30993:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['desires'] += 1
                    elif tmp == -0.57779:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['isa'] += 1
                    elif tmp == -0.14572:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['disable'] += 1
                    elif tmp == -0.30008:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['entails'] += 1
                    elif tmp == 0.015317:
                        flag = 1
                        for i in range(len(index1)):
                            if (index1[i] == tmp1) and (index2[i] == tmp2):
                                flag = 0
                                break
                        if flag:
                            know_num += 1
                            index1.append(tmp1)
                            index2.append(tmp2)
                            relation_dic['antonym'] += 1
                    else:
                        relation_dic['error'] += 1
print(relation_dic)
print("--------------------")
print(know_num)