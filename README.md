# RekNet

Codes for TASLP 2022: Reference Knowledgeable Network for Machine Reading Comprehension

The codes are written based on https://github.com/huggingface/transformers.

The codes are tested with pytorch 1.4.0 and python 3.6.

It is recommended to download the model, config and vocab file and replace the path in ./reknet/transformers/{modeling_electra.py, configuration_electra.py, tokenization_electra.py, modeling_albert.py, configuration_albert.py, tokenization_albert.py}.

We have 'electrasc', 'electratrm', 'electra', 'albertsc', 'alberttrm', 'albertbaseline' six different model_type, which refers to two different types of Attention module and the baseline for ELECTRA and ALBERT baseline.

RekNet can deal with RACE/RACE_middle/RACE_high, DREAM and Cosmos QA task whose task_name is 'race_full'/'racem_full'/'raceh_full', 'dream_full' and 'cosmos_full'.

**Due to the huge size of pre-processed data, we upload it separately, instead of in this code.**

## Data Preparation

Our data is pre-processed and you can download pre-processed data provided by us in future (we recommend this approach which can save a lot of operations and time).

If not, you can also produce the pre-processed data by following steps:

I. Download original RACE (http://www.cs.cmu.edu/~glai1/data/race/), DREAM (https://github.com/nlpdata/dream/tree/master/data) data and Cosmos QA (https://github.com/wilburOne/cosmosqa/tree/master/data/) data online and save them at './reference_finder/{race_data, dream_data, cosmos_data/}'

II. Train or download the 'model.bin' for Reference Finder which is trained on SQuAD and save it at './reference_finder/model/'.
 
III. Enter './reference_finder/' and run following scripts in order:

```bash
python race_to_json.py
CUDA_VISIBLE_DEVICES=0,1 python get_reference_dream.py
CUDA_VISIBLE_DEVICES=0,1 python get_reference_race.py
CUDA_VISIBLE_DEVICES=0,1 python get_reference_cosmos.py
```

After that, rename '\*_reference.json' to '\*.json' and delete other files or directories in './reference_finder/{race_data, dream_data, cosmos_data}', then move these three directories to './knowledge_adapter/'.

IV. Download ConceptNet 5.7.0 (https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz) and GloVe 100d files (https://nlp.stanford.edu/projects/glove/) online and save them at './knowledge_adapter/conceptnet/'

V. Enter './knowledge_adapter/conceptnet/' and run following scripts in order:

```bash
python get_conceptnet.py
python get_merge_conceptnet.py
python get_relation_vector.py
python get_concepts.py {train_dream, dev_dream, test_dream, train_high, dev_high, test_high, train_middle, dev_middle, test_middle}
python get_concepts_cosmos.py {train_cosmos, dev_cosmos, test_cosmos}
python get_knowledge_bash_dream.py generate_bash
python get_knowledge_bash_race.py generate_bash
python get_knowledge_bash_cosmos.py generate_bash
bash *.sh
python get_knowledge_bash_dream.py combine {train_dream, dev_dream, test_dream, train_high, dev_high, test_high, train_middle, dev_middle, test_middle}
python get_knowledge_bash_cosmos.py {train_cosmos, dev_cosmos, test_cosmos}
python get_knowledge_vector_bash_dream.py generate_bash
python get_knowledge_vector_bash_race.py generate_bash
python get_knowledge_vector_bash_cosmos.py generate_bash
bash *.sh
python get_knowledge_vector_bash_dream.py combine {train_dream, dev_dream, test_dream, train_high, dev_high, test_high, train_middle, dev_middle, test_middle}
python get_knowledge_vector_bash_cosmos.py {train_cosmos, dev_cosmos, test_cosmos}
```

After that, rename '\*_knowledge_vector.json' to '\*.json' and delete other files in './knowledge_adapter/{race_data, dream_data, cosmos_data/}', then move these three directories to './reknet/'. Then you can get the pre-processed data, which is same as the data we provided.

## Run RekNet

I. Save the pre-processed data you have got to './reknet/', and the structure of './reknet/' is:

```bash
--examples
    run_multiple_choice.py
    utils_multiple_choice.py
    run_multiple_choice_cosmos.py
    utils_multiple_choice_cosmos.py
--transformers
    ...
--race_data
    train_high.json
    dev_high.json
    test_high.json
    train_middle.json
    dev_middle.json
    test_middle.json
--dream_data
    train.json
    dev.json
    test.json
--cosmos_data
    train.json
    dev.json
    test.json
```

II. Enter './reknet/'. To run ALBERTSC on DREAM dataset, an example script is:

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_multiple_choice.py \
--do_lower_case \
--do_train \
--do_eval \
--do_test \
--overwrite_output \
--overwrite_cache \
--eval_all_checkpoints \
--task_name dream_full \
--per_gpu_eval_batch_size=12 \
--logging_steps 1 \
--max_seq_length 512 \
--max_ref_length 512 \
--model_type albertsc \
--model_name_or_path albert-base-v2 \
--data_dir ../data/ \
--learning_rate 5e-6 \
--num_train_epochs 2 \
--output_dir albertsc_base_dream \
--per_gpu_train_batch_size=1 \
--gradient_accumulation_steps 1 \
--warmup_steps 50 \
--save_steps 382
```

Performance outputs of checkpoints will be saved in 'my_eval_results.txt'.
To get the test labels of cosmos, replace 'run_multiple_choice.py' by 'run_multiple_choice_cosmos.py', and you will get the a file named 'cosmos_test_answers.lst' as the prediction answers.
