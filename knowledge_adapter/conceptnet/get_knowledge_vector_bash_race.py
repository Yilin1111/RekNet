import sys
import json
import configparser
import jsbeautifier


def generate_bash():
    with open("train_m0.sh", 'w') as f:
        for i in range(0, 25):
            f.write("python get_knowledge_vector.py train_middle %d &\n" % i)
        f.write('wait')
    with open("train_m1.sh", 'w') as f:
        for i in range(25, 50):
            f.write("python get_knowledge_vector.py train_middle %d &\n" % i)
        f.write('wait')
    with open("train_m2.sh", 'w') as f:
        for i in range(50, 75):
            f.write("python get_knowledge_vector.py train_middle %d &\n" % i)
        f.write('wait')
    with open("train_m3.sh", 'w') as f:
        for i in range(75, 100):
            f.write("python get_knowledge_vector.py train_middle %d &\n" % i)
        f.write('wait')

    with open("train_h0.sh", 'w') as f:
        for i in range(0, 25):
            f.write("python get_knowledge_vector.py train_high %d &\n" % i)
        f.write('wait')
    with open("train_h1.sh", 'w') as f:
        for i in range(25, 50):
            f.write("python get_knowledge_vector.py train_high %d &\n" % i)
        f.write('wait')
    with open("train_h2.sh", 'w') as f:
        for i in range(50, 75):
            f.write("python get_knowledge_vector.py train_high %d &\n" % i)
        f.write('wait')
    with open("train_h3.sh", 'w') as f:
        for i in range(75, 100):
            f.write("python get_knowledge_vector.py train_high %d &\n" % i)
        f.write('wait')

    with open("dev_m0.sh", 'w') as f:
        for i in range(0, 25):
            f.write("python get_knowledge_vector.py dev_middle %d &\n" % i)
        f.write('wait')
    with open("dev_m1.sh", 'w') as f:
        for i in range(25, 50):
            f.write("python get_knowledge_vector.py dev_middle %d &\n" % i)
        f.write('wait')
    with open("dev_m2.sh", 'w') as f:
        for i in range(50, 75):
            f.write("python get_knowledge_vector.py dev_middle %d &\n" % i)
        f.write('wait')
    with open("dev_m3.sh", 'w') as f:
        for i in range(75, 100):
            f.write("python get_knowledge_vector.py dev_middle %d &\n" % i)
        f.write('wait')

    with open("dev_h0.sh", 'w') as f:
        for i in range(0, 25):
            f.write("python get_knowledge_vector.py dev_high %d &\n" % i)
        f.write('wait')
    with open("dev_h1.sh", 'w') as f:
        for i in range(25, 50):
            f.write("python get_knowledge_vector.py dev_high %d &\n" % i)
        f.write('wait')
    with open("dev_h2.sh", 'w') as f:
        for i in range(50, 75):
            f.write("python get_knowledge_vector.py dev_high %d &\n" % i)
        f.write('wait')
    with open("dev_h3.sh", 'w') as f:
        for i in range(75, 100):
            f.write("python get_knowledge_vector.py dev_high %d &\n" % i)
        f.write('wait')

    with open("test_m0.sh", 'w') as f:
        for i in range(0, 25):
            f.write("python get_knowledge_vector.py test_middle %d &\n" % i)
        f.write('wait')
    with open("test_m1.sh", 'w') as f:
        for i in range(25, 50):
            f.write("python get_knowledge_vector.py test_middle %d &\n" % i)
        f.write('wait')
    with open("test_m2.sh", 'w') as f:
        for i in range(50, 75):
            f.write("python get_knowledge_vector.py test_middle %d &\n" % i)
        f.write('wait')
    with open("test_m3.sh", 'w') as f:
        for i in range(75, 100):
            f.write("python get_knowledge_vector.py test_middle %d &\n" % i)
        f.write('wait')

    with open("test_h0.sh", 'w') as f:
        for i in range(0, 25):
            f.write("python get_knowledge_vector.py test_high %d &\n" % i)
        f.write('wait')
    with open("test_h1.sh", 'w') as f:
        for i in range(25, 50):
            f.write("python get_knowledge_vector.py test_high %d &\n" % i)
        f.write('wait')
    with open("test_h2.sh", 'w') as f:
        for i in range(50, 75):
            f.write("python get_knowledge_vector.py test_high %d &\n" % i)
        f.write('wait')
    with open("test_h3.sh", 'w') as f:
        for i in range(75, 100):
            f.write("python get_knowledge_vector.py test_high %d &\n" % i)
        f.write('wait')


def combine():
    opts = jsbeautifier.default_options()
    opts.indent_size = 2
    final_json = []
    filename = sys.argv[2]

    for i in range(100):
        with open(config["paths"][filename] + "_knowledge_vector_%d.json" % i, 'r') as fp:
            tmp_list = json.load(fp)
        final_json += tmp_list
    with open(config["paths"][filename] + "_knowledge_vector.json", 'w') as fp:
       json.dump(final_json, fp)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("paths.cfg")
    globals()[sys.argv[1]]()
