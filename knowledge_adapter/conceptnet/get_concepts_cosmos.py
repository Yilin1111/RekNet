import configparser
import json
import spacy
from spacy.matcher import Matcher
import sys
from tqdm import tqdm

blacklist = {"-PRON-", "actually", "likely", "possibly", "want", "make", "my", "someone", "sometimes_people", "one",
             "sometimes", "would", "want_to", "something", "sometimes", "everybody", "somebody", "could", "could_be"}
config = configparser.ConfigParser()
config.read("paths.cfg")
with open(config["paths"]["concept_list"], "r", encoding="utf8") as f:
    cpnet_vocab = [l.strip() for l in list(f.readlines())]
cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]


def lemmatize(nlp, concept):
    doc = nlp(concept.replace("_"," "))
    lcs = set()
    lcs.add("_".join([token.lemma_ for token in doc])) # all lemma
    return lcs


def load_matcher(nlp):
    with open(config["paths"]["matcher_patterns"], "r", encoding="utf8") as f:
        all_patterns = json.load(f)
    matcher = Matcher(nlp.vocab)
    for concept, pattern in tqdm(all_patterns.items(), desc="Adding patterns to Matcher."):
        matcher.add(concept, None, pattern)
    return matcher


def ground_mentioned_concepts(nlp, matcher, s, ans=""):
    s = s.lower()
    doc = nlp(s)
    matches = matcher(doc)
    mentioned_concepts = set()
    span_to_concepts = {}

    for match_id, start, end in matches:
        span = doc[start:end].text  # the matched span
        if len(set(span.split(" ")).intersection(set(ans.split(" ")))) > 0:
            continue
        original_concept = nlp.vocab.strings[match_id]
        if len(original_concept.split("_")) == 1:
            original_concept = list(lemmatize(nlp, original_concept))[0]
        if span not in span_to_concepts:
            span_to_concepts[span] = set()
        span_to_concepts[span].add(original_concept)

    for span, concepts in span_to_concepts.items():
        concepts_sorted = list(concepts)
        concepts_sorted.sort(key=len)
        shortest = concepts_sorted[0:3]
        for c in shortest:
            if c in blacklist:
                continue
            lcs = lemmatize(nlp, c)
            intersect = lcs.intersection(shortest)
            if len(intersect) > 0:
                mentioned_concepts.add(list(intersect)[0])
            else:
                mentioned_concepts.add(c)
    return mentioned_concepts


def hard_ground(nlp, sent):
    global cpnet_vocab
    sent = sent.lower()
    doc = nlp(sent)
    res = set()
    for t in doc:
        if t.lemma_ in cpnet_vocab:
            res.add(t.lemma_)
    sent = "_".join([t.text for t in doc])
    if sent in cpnet_vocab:
        res.add(sent)
    return res


def match_mentioned_concepts(nlp, matcher, sent, choices):
    res_question_concepts = set()
    res_answer_concepts = []
    for choice in choices:
        all_concepts = ground_mentioned_concepts(nlp, matcher, sent, choice)
        answer_concepts = ground_mentioned_concepts(nlp, matcher, choice)
        question_concepts = all_concepts - answer_concepts
        if len(question_concepts) == 0:
            question_concepts = hard_ground(nlp, sent)
        if len(answer_concepts) == 0:
            answer_concepts = hard_ground(nlp, choice)
        res_question_concepts = res_question_concepts | question_concepts
        res_answer_concepts.append(list(answer_concepts))
    return list(res_question_concepts), res_answer_concepts


def process(filename):
    with open(config["paths"][filename] + ".json", 'r') as reader:
        input_data = json.load(reader)
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    matcher = load_matcher(nlp)
    res = []
    for entry in tqdm(input_data):
        context = entry['context']
        question = entry['question']
        choices_list = entry['choices_list']
        answer = entry['answer']
        reference = entry['reference']
        ori_question = entry['ori_question']
        entry_id = entry['entry_id']
        sent = reference
        q_concepts, a_concepts = match_mentioned_concepts(nlp, matcher, sent=sent, choices=choices_list)
        res.append({"context": context, "question": question, "choices_list": choices_list, "answer": answer,
                    "reference": reference, "ori_question": ori_question, "q_concepts": q_concepts,
                    "a_concepts": a_concepts, "entry_id": entry_id})
    print("Writing to file...")
    with open(config["paths"][filename] + "_concept.json", 'w') as writer:
        json.dump(res, writer)


if __name__ == "__main__":
    process(sys.argv[1])
