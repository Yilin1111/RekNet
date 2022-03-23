import configparser

config = configparser.ConfigParser()
config.read("paths.cfg")
merge_context = []

relation_mapping = {}
with open(config["paths"]["merge_relation"], encoding="utf-8") as f1:
    for line in f1.readlines():
        line_list = line.strip().split("/")
        rel = line_list[0]
        for l in line_list:
            if l.startswith("*"):
                relation_mapping[l[1:]] = rel
            else:
                relation_mapping[l] = rel

with open(config["paths"]["conceptnet_en"], encoding="utf-8") as f2:
    for line in f2.readlines():
        line_list = line.split("\t")
        if line_list[0] not in relation_mapping:
            continue
        if line_list[0].startswith("*"):
            relation = relation_mapping[line_list[0][1:]]
            subject = line_list[2]
            object = line_list[1]
        else:
            relation = relation_mapping[line_list[0]]
            subject = line_list[1]
            object = line_list[2]
        weight = line_list[3]
        merge_context.append("\t".join([relation, subject, object, str(weight)]))

with open(config["paths"]["conceptnet_merge"], "w", encoding="utf-8") as f:
    f.write("".join(merge_context))
