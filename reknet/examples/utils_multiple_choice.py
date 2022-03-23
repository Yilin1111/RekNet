# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """

from __future__ import absolute_import, division, print_function

import logging
import os
import sys
from io import open
import json
import csv
import glob
import tqdm
from typing import List
from transformers import PreTrainedTokenizer
import random
# from mctest import parse_mc
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None, references=None,
                 knowledge_vector_lists=None, nsp_label=None, context_sents=None):
        """Constructs a InputExample.
        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label
        self.references = references
        self.knowledge_vector_lists = knowledge_vector_lists


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label,
                 ref_choices_features,
                 knowledge_vector_lists=None,
                 pq_end_pos=None):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
            }
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.ref_choices_features = [
            {
                'ref_input_ids': input_ids,
                'ref_input_mask': input_mask,
                'ref_segment_ids': segment_ids,
            }
            for input_ids, input_mask, segment_ids in ref_choices_features
        ]
        self.knowledge_vector_lists = [
            {
                'knowledge_vectors': knowledge_vectors,
            }
            for knowledge_vectors in knowledge_vector_lists
        ]
        # self.knowledge_vector_lists = knowledge_vector_lists
        self.label = label
        self.pq_end_pos = pq_end_pos


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""
    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(data_dir, "test")

    def _create_examples(self, data_dir: str, type: str):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(data_dir + "/" + type + "_high.json", "r") as fh:
            data = json.load(fh)
            for i in range(len(data)):
                # context, reference, question + common choice, answer
                d = [data[i]["context"].lower(), data[i]["reference"].lower(), data[i]["question"].lower(),
                     data[i]["answer"].lower()]
                # choice
                for j in range(len(data[i]["choices_list"])):
                    d += [data[i]["choices_list"][j].lower()]
                d.append(data[i]["knowledge_vector"])
                for k in range(4):
                    if d[4 + k] == d[3]:
                        answer = str(k)
                        break
                label = answer
                examples.append(
                    InputExample(example_id=data[i]["entry_id"], contexts=[d[0], d[0], d[0], d[0]],
                                 references=[d[1], d[1], d[1], d[1]], question=d[2],
                                 endings=[d[4], d[5], d[6], d[7]], label=label,
                                 knowledge_vector_lists=[d[8], d[8], d[8], d[8]]))

        with open(data_dir + "/" + type + "_middle.json", "r") as fm:
            data = json.load(fm)
            for i in range(len(data)):
                # context, reference, question + common choice, answer
                d = [data[i]["context"].lower(), data[i]["reference"].lower(), data[i]["question"].lower(),
                     data[i]["answer"].lower()]
                # choice
                for j in range(len(data[i]["choices_list"])):
                    d += [data[i]["choices_list"][j].lower()]
                d.append(data[i]["knowledge_vector"])
                for k in range(4):
                    if d[4 + k] == d[3]:
                        answer = str(k)
                        break
                label = answer
                examples.append(
                    InputExample(example_id=data[i]["entry_id"], contexts=[d[0], d[0], d[0], d[0]],
                                 references=[d[1], d[1], d[1], d[1]], question=d[2],
                                 endings=[d[4], d[5], d[6], d[7]], label=label,
                                 knowledge_vector_lists=[d[8], d[8], d[8], d[8]]))
        return examples


class RaceHProcessor(DataProcessor):
    """Processor for the RACE data set."""
    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(data_dir, "test")

    def _create_examples(self, data_dir: str, type: str):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(data_dir + "/" + type + "_high.json", "r") as fh:
            data = json.load(fh)
            for i in range(len(data)):
                # context, reference, question + common choice, answer
                d = [data[i]["context"].lower(), data[i]["reference"].lower(), data[i]["question"].lower(),
                     data[i]["answer"].lower()]
                # choice
                for j in range(len(data[i]["choices_list"])):
                    d += [data[i]["choices_list"][j].lower()]
                d.append(data[i]["knowledge_vector"])
                for k in range(4):
                    if d[4 + k] == d[3]:
                        answer = str(k)
                        break
                label = answer
                examples.append(
                    InputExample(example_id=data[i]["entry_id"], contexts=[d[0], d[0], d[0], d[0]],
                                 references=[d[1], d[1], d[1], d[1]], question=d[2],
                                 endings=[d[4], d[5], d[6], d[7]], label=label,
                                 knowledge_vector_lists=[d[8], d[8], d[8], d[8]]))
        return examples


class RaceMProcessor(DataProcessor):
    """Processor for the RACE data set."""
    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(data_dir, "test")

    def _create_examples(self, data_dir: str, type: str):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(data_dir + "/" + type + "_middle.json", "r") as fm:
            data = json.load(fm)
            for i in range(len(data)):
                # context, reference, question + common choice, answer
                d = [data[i]["context"].lower(), data[i]["reference"].lower(), data[i]["question"].lower(),
                     data[i]["answer"].lower()]
                # choice
                for j in range(len(data[i]["choices_list"])):
                    d += [data[i]["choices_list"][j].lower()]
                d.append(data[i]["knowledge_vector"])
                for k in range(4):
                    if d[4 + k] == d[3]:
                        answer = str(k)
                        break
                label = answer
                examples.append(
                    InputExample(example_id=data[i]["entry_id"], contexts=[d[0], d[0], d[0], d[0]],
                                 references=[d[1], d[1], d[1], d[1]], question=d[2],
                                 endings=[d[4], d[5], d[6], d[7]], label=label,
                                 knowledge_vector_lists=[d[8], d[8], d[8], d[8]]))
        return examples


class SwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        if type == "train" and lines[0][-1] != 'label':
            raise ValueError(
                "For training, the input file must contain a label column."
            )

        examples = [
            InputExample(
                example_id=line[2],
                question=line[5],  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[4], line[4], line[4], line[4]],
                endings=[line[7], line[8], line[9], line[10]],
                label=line[11]
            ) for line in lines[1:]  # we skip the line with the column names
        ]

        return examples


class DreamProcessor(DataProcessor):
    """Processor for the DREAM with knowledge from reference(has reference and context) data set, combine knowledge in other part."""

    def __init__(self):
        self.data_pos = {"train": 0, "dev": 1, "test": 2}
        self.D = [[], [], []]

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _create_examples(self, data_dir: str, type: str):
        """Creates examples for the training and dev sets."""
        if len(self.D[self.data_pos[type]]) == 0:
            random.seed(42)
            for sid in range(3):
                with open([data_dir + "/" + "train.json", data_dir + "/" + "dev.json",
                           data_dir + "/" + "test.json"][sid], "r") as f:
                    data = json.load(f)
                    if sid == 0:
                        random.shuffle(data)
                    for i in range(len(data)):
                        # context, reference, question + common choice, answer
                        d = [data[i]["context"].lower(), data[i]["reference"].lower(), data[i]["question"].lower(),
                             data[i]["answer"].lower()]
                        # choice
                        for j in range(len(data[i]["choices_list"])):
                            d += [data[i]["choices_list"][j].lower()]
                        d.append(data[i]["knowledge_vector"])
                        self.D[sid] += [d]

        data = self.D[self.data_pos[type]]
        examples = []
        for (i, d) in enumerate(data):
            for k in range(3):
                if data[i][4 + k] == data[i][3]:
                    answer = str(k)
            label = answer
            guid = "%s-%s-%s" % (type, i, k)
            examples.append(
                InputExample(example_id=guid, contexts=[data[i][0], data[i][0], data[i][0]],
                             references=[data[i][1], data[i][1], data[i][1]], question=data[i][2],
                             endings=[data[i][4], data[i][5], data[i][6]], label=label,
                             knowledge_vector_lists=[data[i][7], data[i][7], data[i][7]]))
        return examples


class CosmosProcessor(DataProcessor):
    """Processor for the Cosmos with knowledge from reference(has reference and context) data set, combine knowledge in other part."""

    def __init__(self):
        self.data_pos = {"train": 0, "dev": 1, "test": 2}
        self.D = [[], [], []]

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _create_examples(self, data_dir: str, type: str):
        """Creates examples for the training and dev sets."""
        if len(self.D[self.data_pos[type]]) == 0:
            random.seed(42)
            for sid in range(3):
                with open([data_dir + "/" + "train.json", data_dir + "/" + "dev.json",
                           data_dir + "/" + "test.json"][sid], "r") as f:
                    data = json.load(f)
                    if sid == 0:
                        random.shuffle(data)
                    for i in range(len(data)):
                        # context, reference, question + common choice, answer
                        d = [data[i]["context"].lower(), data[i]["reference"].lower(), data[i]["question"].lower(),
                             data[i]["answer"].lower()]
                        # choice
                        for j in range(len(data[i]["choices_list"])):
                            d += [data[i]["choices_list"][j].lower()]
                        d.append(data[i]["knowledge_vector"])
                        self.D[sid] += [d]

        data = self.D[self.data_pos[type]]
        examples = []
        for (i, d) in enumerate(data):
            answer = '0'
            for k in range(4):
                if data[i][4 + k] == data[i][3]:
                    answer = str(k)
            label = answer
            guid = "%s-%s-%s" % (type, i, k)
            examples.append(
                InputExample(example_id=guid, contexts=[data[i][0], data[i][0], data[i][0], data[i][0]],
                             references=[data[i][1], data[i][1], data[i][1], data[i][1]], question=data[i][2],
                             endings=[data[i][4], data[i][5], data[i][6], data[i][7]], label=label,
                             knowledge_vector_lists=[data[i][8], data[i][8], data[i][8], data[i][8]]))
        return examples


class MctestProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def __init__(self):
        self.data_pos = {"train": 0, "dev": 1, "test": 2}
        self.D = [[], [], []]

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(data_dir, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _create_examples(self, data_dir: str, type: str):
        """Creates examples for the training and dev sets."""
        if "mc500" in data_dir:
            pref = "mc500."
        if "mc160" in data_dir:
            pref = "mc160."
        article, question, ct1, ct2, ct3, ct4, y, q_id = parse_mc(os.path.join(data_dir, pref + type + ".tsv"),
                                                                  os.path.join(data_dir, pref + type + ".ans"))
        examples = []
        for i, (s1, s2, s3, s4, s5, s6, s7, s8), in enumerate(zip(article, question, ct1, ct2, ct3, ct4, y, q_id)):
            examples.append(
                InputExample(example_id=s8, contexts=[s1, s1, s1, s1], question=s2, endings=[s3, s4, s5, s6],
                             label=str(s7)))
        return examples


def read_race(path):
    with open(path, 'r', encoding='utf_8') as f:
        data_all = json.load(f)
        article = []
        question = []
        st = []
        ct1 = []
        ct2 = []
        ct3 = []
        ct4 = []
        y = []
        q_id = []
        for instance in data_all:

            ct1.append(' '.join(instance['options'][0]))
            ct2.append(' '.join(instance['options'][1]))
            ct3.append(' '.join(instance['options'][2]))
            ct4.append(' '.join(instance['options'][3]))
            question.append(' '.join(instance['question']))
            # q_id.append(instance['q_id'])
            q_id.append(0)
            art = instance['article']
            l = []
            for i in art: l += i
            article.append(' '.join(l))
            # article.append(' '.join(instance['article']))
            y.append(instance['ground_truth'])
        return article, question, ct1, ct2, ct3, ct4, y, q_id


class ArcProcessor(DataProcessor):
    """Processor for the ARC data set (request from allennlp)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        # There are two types of labels. They should be normalized
        def normalize(truth):
            if truth in "ABCD":
                return ord(truth) - ord("A")
            elif truth in "1234":
                return int(truth) - 1
            else:
                logger.info("truth ERROR! %s", str(truth))
                return None

        examples = []
        three_choice = 0
        four_choice = 0
        five_choice = 0
        other_choices = 0
        # we deleted example which has more than or less than four choices
        for line in tqdm.tqdm(lines, desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            if len(data_raw["question"]["choices"]) == 3:
                three_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) == 5:
                five_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) != 4:
                other_choices += 1
                continue
            four_choice += 1
            truth = str(normalize(data_raw["answerKey"]))
            assert truth != "None"
            question_choices = data_raw["question"]
            question = question_choices["stem"]
            id = data_raw["id"]
            options = question_choices["choices"]
            if len(options) == 4:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[options[0]["para"].replace("_", ""), options[1]["para"].replace("_", ""),
                                  options[2]["para"].replace("_", ""), options[3]["para"].replace("_", "")],
                        endings=[options[0]["text"], options[1]["text"], options[2]["text"], options[3]["text"]],
                        label=truth))

        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info("len examples: %s}", str(len(examples)))
        logger.info("Three choices: %s", str(three_choice))
        logger.info("Five choices: %s", str(five_choice))
        logger.info("Other choices: %s", str(other_choices))
        logger.info("four choices: %s", str(four_choice))

        return examples


def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_length: int,
        max_ref_length: int,
        tokenizer: PreTrainedTokenizer,
        pad_token_segment_id=0,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
        truncation_strategy='longest_first',
        is_full=False
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    if is_full:  # add reference
        for (ex_index, example) in enumerate(tqdm.tqdm(examples, desc="convert examples to features")):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))
            choices_features = []
            ref_choices_features = []
            all_knowledge_vector_lists = []
            # record end positions of two parts which need interaction such as Passage and Question, for later separating them
            pq_end_pos = []
            for ending_idx, (context, reference, ending, knowledge_vector_lists) in enumerate(zip(
                    example.contexts, example.references, example.endings, example.knowledge_vector_lists)):
                text_a = context  # 每一个QA对的context
                text_b = example.question + " " + ending  # 每一个QA对
                text_d = reference  # 每一个QA对的reference
                special_tok_len = 3  # [CLS] [SEP] [SEP]
                sep_tok_len = 1  # [SEP]
                t_q_len = len(tokenizer.tokenize(example.question))
                t_b_len = len(tokenizer.tokenize(text_b))
                t_o_len = t_b_len - t_q_len
                context_max_len = max_length - special_tok_len - t_q_len - t_o_len
                reference_max_len = max_ref_length - special_tok_len - t_q_len - t_o_len
                t_c_len = len(tokenizer.tokenize(context))
                t_r_len = len(tokenizer.tokenize(reference))
                if t_b_len >= max_ref_length - special_tok_len:
                    truncation_strategy = "longest_first"
                if t_c_len > context_max_len:
                    t_c_len = context_max_len
                if t_r_len > reference_max_len:
                    t_r_len = reference_max_len
                assert (t_q_len + t_o_len + t_c_len <= max_length)
                assert (t_q_len + t_o_len + t_r_len <= max_ref_length)
                inputs = tokenizer.encode_plus(
                    text_a,
                    text_b,
                    add_special_tokens=True,
                    max_length=max_length,
                    truncation_strategy=truncation_strategy
                )
                ref_inputs = tokenizer.encode_plus(
                    text_d,
                    text_b,
                    add_special_tokens=True,
                    max_length=max_ref_length,
                    truncation_strategy=truncation_strategy
                )
                input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
                ref_input_ids, ref_token_type_ids = ref_inputs["input_ids"], ref_inputs["token_type_ids"]
                assert (len(input_ids[t_c_len + t_q_len + t_o_len:]) == special_tok_len)
                assert (len(ref_input_ids[t_r_len + t_q_len + t_o_len:]) == special_tok_len)
                # [CLS] CONTEXT [SEP] QUESTION OPTION [SEP]
                t_pq_end_pos = [1 + t_c_len - 1, 1 + t_c_len + sep_tok_len + t_q_len + t_o_len - 1]
                pq_end_pos.append(t_pq_end_pos)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
                ref_attention_mask = [1 if mask_padding_with_zero else 0] * len(ref_input_ids)
                pad_token = tokenizer.pad_token_id

                # Zero-pad up to the sequence length.
                padding_length = max_length - len(input_ids)
                ref_padding_length = max_ref_length - len(ref_input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    ref_input_ids = ([pad_token] * ref_padding_length) + ref_input_ids
                    attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                    ref_attention_mask = ([
                                              0 if mask_padding_with_zero else 1] * ref_padding_length) + ref_attention_mask
                    token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
                    ref_token_type_ids = ([pad_token_segment_id] * ref_padding_length) + ref_token_type_ids
                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    ref_input_ids = ref_input_ids + ([pad_token] * ref_padding_length)
                    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    ref_attention_mask = ref_attention_mask + (
                            [0 if mask_padding_with_zero else 1] * ref_padding_length)
                    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
                    ref_token_type_ids = ref_token_type_ids + ([pad_token_segment_id] * ref_padding_length)

                assert len(input_ids) == max_length
                assert len(attention_mask) == max_length
                assert len(token_type_ids) == max_length
                assert len(ref_input_ids) == max_ref_length
                assert len(ref_attention_mask) == max_ref_length
                assert len(ref_token_type_ids) == max_ref_length
                blank_list = [[0] * 100] * 4
                # 每个问题三元组上限15个（一个选项5个）
                knowledge_vector_lists = knowledge_vector_lists[0]
                if len(knowledge_vector_lists) < 15:
                    for i in range(15 - len(knowledge_vector_lists)):
                        knowledge_vector_lists.append(blank_list)
                else:
                    knowledge_vector_lists = knowledge_vector_lists[:15]
                choices_features.append((input_ids, attention_mask, token_type_ids))
                ref_choices_features.append((ref_input_ids, ref_attention_mask, ref_token_type_ids))
                all_knowledge_vector_lists.append(knowledge_vector_lists)

            label = label_map[example.label]
            features.append(
                InputFeatures(
                    example_id=example.example_id,
                    choices_features=choices_features,
                    ref_choices_features=ref_choices_features,
                    knowledge_vector_lists=all_knowledge_vector_lists,
                    label=label,
                    pq_end_pos=pq_end_pos
                )
            )
    else:
        for (ex_index, example) in enumerate(tqdm.tqdm(examples, desc="convert examples to features")):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))
            choices_features = []
            all_knowledge_vector_lists = []
            # record end positions of two parts which need interaction such as Passage and Question, for later separating them
            pq_end_pos = []
            for ending_idx, (context, ending, knowledge_vector_lists) in enumerate(zip(
                    example.contexts, example.endings, example.knowledge_vector_lists)):
                text_a = context  # 每一个QA对的context
                text_b = example.question + " " + ending  # 每一个QA对
                special_tok_len = 3  # [CLS] [SEP] [SEP]
                sep_tok_len = 1  # [SEP]
                t_q_len = len(tokenizer.tokenize(example.question))
                # t_o_len=len(tokenizer.tokenize(ending)) # 直接计算会出错，单独对option分词和拼接question再分词，结果不同
                t_o_len = len(tokenizer.tokenize(text_b)) - t_q_len
                context_max_len = max_length - special_tok_len - t_q_len - t_o_len
                t_c_len = len(tokenizer.tokenize(context))
                if t_c_len > context_max_len:
                    t_c_len = context_max_len
                assert (t_q_len + t_o_len + t_c_len <= max_length)
                inputs = tokenizer.encode_plus(
                    text_a,
                    text_b,
                    add_special_tokens=True,
                    max_length=max_length,
                    truncation_strategy=truncation_strategy
                )
                input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
                assert (len(input_ids[t_c_len + t_q_len + t_o_len:]) == special_tok_len)
                # [CLS] CONTEXT [SEP] QUESTION OPTION [SEP]
                t_pq_end_pos = [1 + t_c_len - 1, 1 + t_c_len + sep_tok_len + t_q_len + t_o_len - 1]
                pq_end_pos.append(t_pq_end_pos)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
                pad_token = tokenizer.pad_token_id

                # Zero-pad up to the sequence length.
                padding_length = max_length - len(input_ids)
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                    token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

                assert len(input_ids) == max_length
                assert len(attention_mask) == max_length
                assert len(token_type_ids) == max_length
                blank_list = [[0] * 100] * 4
                knowledge_vector_lists = knowledge_vector_lists[0]
                # 每个问题三元组上限15个（一个选项5个）
                if len(knowledge_vector_lists) < 15:
                    for i in range(15 - len(knowledge_vector_lists)):
                        knowledge_vector_lists.append(blank_list)
                else:
                    knowledge_vector_lists = knowledge_vector_lists[:15]
                choices_features.append((input_ids, attention_mask, token_type_ids))
                all_knowledge_vector_lists.append(knowledge_vector_lists)

            label = label_map[example.label]
            features.append(
                InputFeatures(
                    example_id=example.example_id,
                    choices_features=choices_features,
                    ref_choices_features=choices_features,
                    knowledge_vector_lists=all_knowledge_vector_lists,
                    label=label,
                    pq_end_pos=pq_end_pos
                )
            )
    return features


processors = {
    "race_full": RaceProcessor,
    "racem_full": RaceMProcessor,
    "raceh_full": RaceHProcessor,
    "swag": SwagProcessor,
    "arc": ArcProcessor,
    "dream_full": DreamProcessor,
    "cosmos_full": CosmosProcessor,
    "mctest": MctestProcessor
}

MULTIPLE_CHOICE_TASKS_NUM_LABELS = {
    "race_full", 4,
    "racem_full", 4,
    "raceh_full", 4,
    "swag", 4,
    "arc", 4,
    "dream_full", 3,
    "cosmos_full", 4,
    "mctest", 4
}
