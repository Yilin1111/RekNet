import configparser
import json


def del_pos(s):
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s


def extract_english():
    config = configparser.ConfigParser()
    config.read("paths.cfg")
    only_english = []
    concept_list = []
    with open(config["paths"]["conceptnet"], encoding="utf8") as f:
        last_head = ""
        last_tail = ""
        for line in f.readlines():
            ls = line.split('\t')
            if ls[2].startswith('/c/en/') and ls[3].startswith('/c/en/'):
                rel = ls[1].split("/")[-1].lower()
                head = del_pos(ls[2]).split("/")[-1].lower()
                tail = del_pos(ls[3]).split("/")[-1].lower()
                if not head.replace("_", "").replace("-", "").isalpha():
                    continue
                if not tail.replace("_", "").replace("-", "").isalpha():
                    continue
                if (last_head == head) and (last_tail == tail):
                    continue
                if head == tail:
                    continue
                last_head = head
                last_tail = tail
                data = json.loads(ls[4])
                only_english.append("\t".join([rel, head, tail, str(data["weight"])]))
                concept_list.append(head)
                concept_list.append(tail)

    with open(config["paths"]["conceptnet_en"], "w", encoding="utf8") as f:
        f.write("\n".join(only_english))
    concept_list = list(set(concept_list))
    concept_list.sort()
    with open(config["paths"]["concept_list"], "w", encoding="utf8") as f:
        f.write("\n".join(concept_list))


if __name__ == "__main__":
    extract_english()
