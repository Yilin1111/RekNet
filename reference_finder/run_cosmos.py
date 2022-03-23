"""
input: cosmos train dataset(json file)
output: cosmos train dataset with reference span(json file)
"""
from __future__ import absolute_import, division, print_function
import argparse
import logging
import random
import timeit
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from transformers import (BertConfig, BertForQuestionAnswering, BertTokenizer, XLMConfig, XLMForQuestionAnswering,
                          XLMTokenizer, XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer,
                          DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer,
                          AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer)
from utils_cosmos import (read_cosmos_examples, convert_examples_to_features,
                         RawResult, write_predictions, RawResultExtended)

logger = logging.getLogger(__name__)
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig)), ())
MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
    'albert': (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def evaluate(args, model, tokenizer, input_file, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, input_file, evaluate=True, output_examples=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    start_time = timeit.default_timer()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1]
                      }
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = None if args.model_type == 'xlm' else batch[2]  # XLM don't use segment_ids
            example_indices = batch[3]
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4], 'p_mask':    batch[5]})
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            if args.model_type in ['xlnet', 'xlm']:
                # XLNet uses a more complex post-processing procedure
                result = RawResultExtended(unique_id            = unique_id,
                                           start_top_log_probs  = to_list(outputs[0][i]),
                                           start_top_index      = to_list(outputs[1][i]),
                                           end_top_log_probs    = to_list(outputs[2][i]),
                                           end_top_index        = to_list(outputs[3][i]),
                                           cls_logits           = to_list(outputs[4][i]))
            else:
                result = RawResult(unique_id    = unique_id,
                                   start_logits = to_list(outputs[0][i]),
                                   end_logits   = to_list(outputs[1][i]))
            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    all_predictions = write_predictions(examples, features, all_results, args.n_best_size,
                    args.max_answer_length, args.do_lower_case, args.verbose_logging,
                    args.version_2_with_negative, args.null_score_diff_threshold)

    return all_predictions, examples


def load_and_cache_examples(args, tokenizer, input_file, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    examples = read_cosmos_examples(input_file)
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            max_seq_length=args.max_seq_length,
                                            doc_stride=args.doc_stride,
                                            max_query_length=args.max_query_length,
                                            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                            pad_token_segment_id=3 if args.model_type in ['xlnet'] else 0,
                                            cls_token_at_end=True if args.model_type in ['xlnet'] else False,
                                            sequence_a_is_doc=True if args.model_type in ['xlnet'] else False)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask)

    if output_examples:
        return dataset, examples, features
    return dataset


parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--model_type", default="bert", type=str,
                help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
parser.add_argument("--model_name_or_path", default=None, type=str,
                help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
parser.add_argument("--output_dir", default=None, type=str,
                help="The output directory where the model checkpoints and predictions will be written.")

## Other parameters
parser.add_argument("--config_name", default="", type=str,
                help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--tokenizer_name", default="", type=str,
                help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--cache_dir", default="", type=str,
                help="Where do you want to store the pre-trained models downloaded from s3")

parser.add_argument('--version_2_with_negative', action='store_true',
                help='If true, the SQuAD examples contain some that do not have an answer.')
parser.add_argument('--null_score_diff_threshold', type=float, default=200,
                help="If null_score - best_non_null is greater than the threshold predict null.")

parser.add_argument("--max_seq_length", default=512, type=int,
                help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                     "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--doc_stride", default=128, type=int,
                help="When splitting up a long document into chunks, how much stride to take between chunks.")
parser.add_argument("--max_query_length", default=64, type=int,
                help="The maximum number of tokens for the question. Questions longer than this will "
                     "be truncated to this length.")
parser.add_argument("--do_train", action='store_true',
                help="Whether to run training.")
parser.add_argument("--do_eval", action='store_true',
                help="Whether to run eval on the dev set.")
parser.add_argument("--evaluate_during_training", action='store_true',
                help="Rul evaluation during training at each logging step.")
parser.add_argument("--do_lower_case", action='store_true',
                help="Set this flag if you are using an uncased model.")

parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                help="The initial learning rate for Adam.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=3.0, type=float,
                help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                help="Linear warmup over warmup_steps.")
parser.add_argument("--n_best_size", default=20, type=int,
                help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
parser.add_argument("--max_answer_length", default=30, type=int,
                help="The maximum length of an answer that can be generated. This is needed because the start "
                     "and end predictions are not conditioned on one another.")
parser.add_argument("--verbose_logging", action='store_true',
                help="If true, all of the warnings related to data processing will be printed. "
                     "A number of warnings are expected for a normal SQuAD evaluation.")

parser.add_argument('--logging_steps', type=int, default=50,
                help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=50,
                help="Save checkpoint every X updates steps.")
parser.add_argument("--eval_all_checkpoints", action='store_true',
                help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
parser.add_argument("--no_cuda", action='store_true',
                help="Whether not to use CUDA when available")
parser.add_argument('--overwrite_output_dir', action='store_true',
                help="Overwrite the content of the output directory")
parser.add_argument('--overwrite_cache', action='store_true',
                help="Overwrite the cached training and evaluation sets")
parser.add_argument('--seed', type=int, default=42,
                help="random seed for initialization")

parser.add_argument("--local_rank", type=int, default=-1,
                help="local_rank for distributed training on gpus")
parser.add_argument('--fp16', action='store_true',
                help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O1',
                help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                     "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
args = parser.parse_args()


class MrcModel(object):
    def __init__(self, checkpoint):
        args.n_gpu = 0
        device = "cuda"
        args.device = device
        args.model_name_or_path = checkpoint
        args.version_2_with_negative = True
        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
        logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                       args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

        # Set seed
        set_seed(args)

        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        self.tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None)

        self.model = model_class.from_pretrained(checkpoint, force_download=True)
        self.model.to(args.device)

    def predict(self, input_file):
        # Reload the model
        # Evaluate
        predictions, examples = evaluate(args, self.model, self.tokenizer, input_file, prefix="")
        for example in examples:
            example.reference_context = predictions[example.question_id]
        return examples
