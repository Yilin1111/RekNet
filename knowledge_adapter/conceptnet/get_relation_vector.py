import configparser


config = configparser.ConfigParser()
config.read("paths.cfg")
relation_set = set()
relation_dic = {}
with open(config["paths"]["conceptnet_merge"], encoding="utf-8") as f1:
    for line in f1.readlines():
        relation_set.add(line.split("\t")[0])

with open(config["paths"]["glove"], encoding="utf-8") as f2:
    for line in f2.readlines():
        information_list = line.split()
        if information_list[0] in relation_set:
            relation_set.remove(information_list[0])
            relation_dic[information_list[0]] = information_list[1:]

with open(config["paths"]["relation_vector"], "w", encoding="utf-8") as f:
    for key in relation_dic.keys():
        f.write(key + " " + " ".join(relation_dic[key]) + "\n")
