import json
from nltk.tokenize import sent_tokenize
from run_cosmos import MrcModel


def write_evidence_to_json(input_file, examples):
    res = []
    for entry in examples:
        context = ''
        question = entry.question_text
        choices_list = entry.choices_list
        answer = entry.answer_text
        evidence = entry.evidence_context
        evidence_sentence = entry.evidence_context
        ori_question = entry.ori_question_text
        entry_id = entry.question_id
        for c in entry.doc_tokens:
            context += (c + ' ')
        all_sentences = sent_tokenize(context)
        for sentence in all_sentences:
            if evidence in sentence:
                evidence_sentence = sentence
        res.append({"context": context, "question": question, "choices_list": choices_list, "answer": answer,
                    "evidence": evidence.strip(), "evidence_sentence": evidence_sentence.strip(),
                    "ori_question": ori_question, "entry_id": entry_id})
    print("Writing to file...")
    with open(input_file[:-6] + "_evidence.json", "w") as writer:
        json.dump(res, writer)
    return 0


def write_test_evidence_to_json(input_file, examples):
    res = []
    for entry in examples:
        context = ''
        question = entry.question_text
        choices_list = entry.choices_list
        evidence = entry.evidence_context
        ori_question = entry.ori_question_text
        entry_id = entry.question_id
        for c in entry.doc_tokens:
            context += (c + ' ')
        res.append({"context": context, "question": question, "choices_list": choices_list, "answer": '',
                    "evidence": evidence.strip(), "ori_question": ori_question, "entry_id": entry_id})
    print("Writing to file...")
    with open(input_file[:-6] + "_evidence.json", "w") as writer:
        json.dump(res, writer)
    return 0


mrc_model = MrcModel("model")
input_files = ["./cosmos_data/train.jsonl", "./cosmos_data/dev.jsonl"]
for input_file in input_files:
    write_evidence_to_json(input_file, mrc_model.predict(input_file))
write_test_evidence_to_json("./cosmos_data/test.jsonl", mrc_model.predict("./cosmos_data/test.jsonl"))
