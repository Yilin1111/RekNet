import json
from run_dream import MrcModel


def write_evidence_to_json(input_file, examples):
    res = []
    for entry in examples:
        context = ''
        question = entry.question_text
        choices_list = entry.choices_list
        answer = entry.answer_text
        reference = entry.reference_context
        ori_question = entry.ori_question_text
        entry_id = entry.question_id
        for c in entry.doc_tokens:
            context += (c + ' ')
        res.append({"context": context, "question": question, "choices_list": choices_list, "answer": answer,
                    "reference": reference.strip(), "ori_question": ori_question, "entry_id": entry_id})
    print("Writing to file...")
    with open(input_file[:-5] + "_references.json", "w") as writer:
        json.dump(res, writer)
    return 0


mrc_model = MrcModel("model")
input_files = ["./dream_data/train.json", "./dream_data/dev.json", "./dream_data/test.json"]
# input_files = ["./dream_data/example.json"]
for input_file in input_files:
    write_evidence_to_json(input_file, mrc_model.predict(input_file))
