import json
from nltk.tokenize import sent_tokenize
from run_race import MrcModel


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
    with open(input_file[:-5] + "_evidence.json", "w") as writer:
        json.dump(res, writer)
    return 0


mrc_model = MrcModel("model")
input_files = ["./race_data/train_high.json", "./race_data/train_middle.json", "./race_data/dev_high.json",
                    "./race_data/dev_middle.json", "./race_data/test_high.json", "./race_data/test_middle.json"]
for input_file in input_files:
    write_evidence_to_json(input_file, mrc_model.predict(input_file))
