import sys
import json
import configparser
import jsbeautifier


def generate_bash():
    with open("train0.sh", 'w') as f:
        for i in range(0, 25):
            f.write("python get_knowledge.py train_cosmos %d &\n" % i)
        f.write('wait')
    with open("train1.sh", 'w') as f:
        for i in range(25, 50):
            f.write("python get_knowledge.py train_cosmos %d &\n" % i)
        f.write('wait')
    with open("train2.sh", 'w') as f:
        for i in range(50, 75):
            f.write("python get_knowledge.py train_cosmos %d &\n" % i)
        f.write('wait')
    with open("train3.sh", 'w') as f:
        for i in range(75, 100):
            f.write("python get_knowledge.py train_cosmos %d &\n" % i)
        f.write('wait')

    with open("dev0.sh", 'w') as f:
        for i in range(0, 25):
            f.write("python get_knowledge.py dev_cosmos %d &\n" % i)
        f.write('wait')
    with open("dev1.sh", 'w') as f:
        for i in range(25, 50):
            f.write("python get_knowledge.py dev_cosmos %d &\n" % i)
        f.write('wait')
    with open("dev2.sh", 'w') as f:
        for i in range(50, 75):
            f.write("python get_knowledge.py dev_cosmos %d &\n" % i)
        f.write('wait')
    with open("dev3.sh", 'w') as f:
        for i in range(75, 100):
            f.write("python get_knowledge.py dev_cosmos %d &\n" % i)
        f.write('wait')

    with open("test0.sh", 'w') as f:
        for i in range(0, 25):
            f.write("python get_knowledge.py test_cosmos %d &\n" % i)
        f.write('wait')
    with open("test1.sh", 'w') as f:
        for i in range(25, 50):
            f.write("python get_knowledge.py test_cosmos %d &\n" % i)
        f.write('wait')
    with open("test2.sh", 'w') as f:
        for i in range(50, 75):
            f.write("python get_knowledge.py test_cosmos %d &\n" % i)
        f.write('wait')
    with open("test3.sh", 'w') as f:
        for i in range(75, 100):
            f.write("python get_knowledge.py test_cosmos %d &\n" % i)
        f.write('wait')


def combine():
    opts = jsbeautifier.default_options()
    opts.indent_size = 2
    final_json = []
    filename = sys.argv[2]

    for i in range(100):
        with open(config["paths"][filename] + "_triple_ids_%d.json" % i, 'r') as fp:
            tmp_list = json.load(fp)
        final_json += tmp_list
    with open(config["paths"][filename] + "_triple_ids.json", 'w') as fp:
       json.dump(final_json, fp)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("paths.cfg")
    globals()[sys.argv[1]]()
