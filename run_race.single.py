# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""BERT finetuning runner."""


import time
import logging
import os
import argparse
import random
import pickle

import csv
import glob
import json
import numpy as np
from numpy.lib.index_tricks import OGridClass
import torch
import torch.nn as nn

from tqdm import tqdm, trange
from apex import amp
from apex.multi_tensor_apply import multi_tensor_applier
from apex.optimizers import FusedAdam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import BertForMultipleChoice
# from pytorch_pretrained_bert.modeling import BertForMultipleChoiceWithMatch
from pytorch_pretrained_bert.optimization import BertAdam
# from pytorch_pretrained_bert.optimization import RAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from pytorch_pretrained_bert.utils import is_main_process

from transformers import AdamW, get_linear_schedule_with_warmup
from multiprocessing import cpu_count
# from transformers import AlbertForMultipleChoice, AlbertTokenizer
from pytorch_pretrained_bert.modeling_albert import AlbertForMultipleChoice, AlbertConfig

from transformers import BertForMultipleChoice
from pytorch_pretrained_bert import tokenization_albert

from tensorboardX import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

OLD_MODE = False  # Don't change
SIMULATE = False  # Don't change


class RaceExample(object):
    """A single training/test example for the RACE dataset."""
    '''
    For RACE dataset:
    race_id: data id
    context_sentence: article
    start_ending: question
    ending_0/1/2/3: option_0/1/2/3
    label: true answer
    '''

    def __init__(self,
                 race_id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label=None):
        self.race_id = race_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"id: {self.race_id}",
            f"article: {self.context_sentence}",
            f"question: {self.start_ending}",
            f"option_0: {self.endings[0]}",
            f"option_1: {self.endings[1]}",
            f"option_2: {self.endings[2]}",
            f"option_3: {self.endings[3]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'doc_len': doc_len,
                'ques_len': ques_len,
                'option_len': option_len
            }
            for _, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len in choices_features
        ]
        self.label = label


# paths is a list containing all paths
def read_race_examples(paths):
    examples = []
    for path in paths:
        filenames = glob.glob(path+"/*txt")
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as fpr:
                data_raw = json.load(fpr)
                article = data_raw['article']
                # for each qn
                for i in range(len(data_raw['answers'])):
                    truth = ord(data_raw['answers'][i]) - ord('A')
                    question = data_raw['questions'][i]
                    options = data_raw['options'][i]
                    examples.append(
                        RaceExample(
                            race_id=filename+'-'+str(i),
                            context_sentence=article,
                            start_ending=question,

                            ending_0=options[0],
                            ending_1=options[1],
                            ending_2=options[2],
                            ending_3=options[3],
                            label=truth))

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, savename):
    """Loads a data file into a list of `InputBatch`s."""

    # RACE is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # The input will be like:
    # [CLS] Article [SEP] Question + Option [SEP]
    # for each option
    #
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    max_option_len = 0
    examples_iter = tqdm(examples, desc="Preprocessing: ", disable=False) if is_main_process() else examples
    for example_index, example in enumerate(examples_iter):
        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            if False:
                # We create a copy of the context tokens in order to be
                # able to shrink it according to ending_tokens
                context_tokens_choice = context_tokens[:]
                ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)

                ending_token = tokenizer.tokenize(ending)
                option_len = len(ending_token)
                ques_len = len(start_ending_tokens)
                doc_len = len(context_tokens_choice)

                # Modifies `context_tokens_choice` and `ending_tokens` in
                # place so that the total length is less than the
                # specified length.  Account for [CLS], [SEP], [SEP] with
                # "- 3"
                _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]  # + start_ending_tokens

            ending_token = tokenizer.tokenize(ending)
            option_len = len(ending_token)
            ques_len = len(start_ending_tokens)

            ending_tokens = start_ending_tokens + ending_token

            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            # ending_tokens = start_ending_tokens + ending_tokens
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)
            doc_len = len(context_tokens_choice)
            if len(ending_tokens) + len(context_tokens_choice) >= max_seq_length - 3:
                ques_len = len(ending_tokens) - option_len

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len))

        label = example.label

        features.append(
            InputFeatures(
                example_id=example.race_id,
                choices_features=choices_features,
                label=label
            )
        )

    with open(savename, "wb") as f:
        pickle.dump(features, f)
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


class GradientClipper:
    """
    Clips gradient norm of an iterable of parameters.
    """

    def __init__(self, max_grad_norm):
        self.max_norm = max_grad_norm
        if multi_tensor_applier.available:
            import amp_C
            self._overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
            self.multi_tensor_scale = amp_C.multi_tensor_scale
        else:
            raise RuntimeError('Gradient clipping requires cuda extensions')

    def step(self, parameters):
        l = [p.grad for p in parameters if p.grad is not None]
        total_norm, _ = multi_tensor_applier(self.multi_tensor_l2norm, self._overflow_buf, [l], False)
        total_norm = total_norm.item()
        if (total_norm == float('inf')):
            return
        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            multi_tensor_applier(self.multi_tensor_scale, self._overflow_buf, [l, l], clip_coef)


def main():
    ete_start = time.time()
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="vocab_file path")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--NEW_MODEL', type=int, default=0)
    parser.add_argument('--USE_ADAM', type=int, default=1)
    parser.add_argument('--dataname', type=str, default="")
    parser.add_argument('--grad_clip', type=int, default=1)
    parser.add_argument('--step', type=int, default=0)

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    if is_main_process():
        logger.info("device: {} ({}), n_gpu: {}, distributed training: {}, 16-bits training: {}".format(device, torch.cuda.get_device_name(0), n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.NEW_MODEL:
        # tokenizer = AlbertTokenizer.from_pretrained(os.path.join("albert_model", "30k-clean.vocab"), do_lower_case=args.do_lower_case)
        tokenizer = tokenization_albert.FullTokenizer(os.path.join("albert_model", "30k-clean.vocab"), do_lower_case=args.do_lower_case, spm_model_file=os.path.join("albert_model", "30k-clean.model"))
    else:
        tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_dir = os.path.join(args.data_dir, 'train')
        train_examples = read_race_examples([os.path.join(train_dir, "high"), os.path.join(train_dir, "middle")])
        num_train_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    if args.NEW_MODEL:
        logger.info("Load new model...")
        config = AlbertConfig.from_pretrained(os.path.join("albert_model", "config.json"), num_labels=4)
        model = AlbertForMultipleChoice.from_pretrained(os.path.join("albert_model", "pytorch_model.bin"), config=config)
    else:
        logger.info("Load old model...")
        model = BertForMultipleChoice.from_pretrained(os.path.join("bert_model"), cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
    model.to(device)

    if OLD_MODE:
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = num_train_steps
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
    else:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if args.USE_ADAM:
            # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False)
        else:
            logger.info("Loading RAdam...")
            optimizer = RAdam(optimizer_grouped_parameters, lr=args.learning_rate)

    global_step = 0
    train_start = time.time()
    writer = SummaryWriter(os.path.join(args.output_dir, "ascxx"))
    if args.do_train:
        train_features = 0
        data_path = f"data{args.dataname}.bin"
        if os.path.exists(data_path):
            with open(data_path, "rb") as f:
                train_features = pickle.load(f)
        else:
            train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length, data_path)
        if SIMULATE:
            logger.info("simulating...")
            train_features = random.sample(train_features, 25000)

        if is_main_process():
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_doc_len = torch.tensor(select_field(train_features, 'doc_len'), dtype=torch.long)
        all_ques_len = torch.tensor(select_field(train_features, 'ques_len'), dtype=torch.long)
        all_option_len = torch.tensor(select_field(train_features, 'option_len'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_doc_len, all_ques_len, all_option_len)

        train_sampler = 0
        if args.local_rank != -1:
            train_sampler = DistributedSampler(train_data)
        else:
            train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data,
                                      sampler=train_sampler,
                                      batch_size=args.train_batch_size,
                                      num_workers=0,
                                      prefetch_factor=2,
                                      pin_memory=True)

        model.train()
        if not OLD_MODE:
            # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            model, optimizer = amp.initialize(model,
                                              optimizers=optimizer,
                                              opt_level="O2",
                                              keep_batchnorm_fp32=False,
                                              loss_scale="dynamic")
            scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_steps * args.warmup_proportion), num_train_steps)
        if OLD_MODE:
            gradClipper = GradientClipper(max_grad_norm=1.0)

        if args.local_rank != -1:
            logger.info("Initializing DistributedDataParallel")
            model = DistributedDataParallel(model,
                                            device_ids=[args.local_rank],
                                            output_device=args.local_rank,
                                            find_unused_parameters=True)
            logger.info("DistributedDataParallel initialized")

        for ep in range(int(args.num_train_epochs)):
            tr_loss = 0
            train_iter = tqdm(train_dataloader, disable=False) if is_main_process() else train_dataloader
            if is_main_process():
                train_iter.set_description("Trianing Epoch: {}/{}".format(ep+1, int(args.num_train_epochs)))
            for step, batch in enumerate(train_iter):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, doc_lens, ques_lens, option_lens = batch
                if args.NEW_MODEL:
                    # outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
                    outputs = model(input_ids, input_mask, segment_ids, labels=label_ids)
                    loss = outputs[0]
                else:
                    outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
                    loss = outputs.loss
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                with open("loss.txt", "a", encoding="utf-8") as f:
                    f.write(f"{loss.item()}\n")

                if not OLD_MODE:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                if args.grad_clip:
                    nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.)

                if OLD_MODE:
                    loss.backward()
                    gradClipper.step(amp.master_params(optimizer))

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if OLD_MODE:
                        # modify learning rate with special warm up BERT uses
                        lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    if not OLD_MODE:
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % 500 == 0 and (args.local_rank == 0 or args.local_rank == -1):
                        dev_dir = os.path.join(args.data_dir, 'dev')
                        dev_set = [os.path.join(dev_dir, "high"), os.path.join(dev_dir, "middle")]

                        eval_examples = read_race_examples(dev_set)
                        eval_features = 0
                        eval_path = f"eval{args.dataname}.bin"
                        if os.path.exists(eval_path):
                            with open(eval_path, "rb") as f:
                                eval_features = pickle.load(f)
                        else:
                            eval_features = convert_examples_to_features(eval_examples,
                                                                         tokenizer,
                                                                         args.max_seq_length,
                                                                         eval_path)
                        eval_features = random.sample(eval_features, 300)
                        logger.info("***** Running evaluation: Dev *****")
                        logger.info("  Num examples = %d", len(eval_examples))
                        logger.info("  Batch size = %d", args.eval_batch_size)
                        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
                        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
                        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
                        all_doc_len = torch.tensor(select_field(eval_features, 'doc_len'), dtype=torch.long)
                        all_ques_len = torch.tensor(select_field(eval_features, 'ques_len'), dtype=torch.long)
                        all_option_len = torch.tensor(select_field(eval_features, 'option_len'), dtype=torch.long)
                        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
                        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_doc_len, all_ques_len, all_option_len)
                        # Run prediction for full data
                        eval_sampler = SequentialSampler(eval_data)
                        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                        model.eval()
                        eval_loss, eval_accuracy = 0, 0
                        nb_eval_steps, nb_eval_examples = 0, 0
                        for step, batch in enumerate(eval_dataloader):
                            batch = tuple(t.to(device) for t in batch)
                            input_ids, input_mask, segment_ids, label_ids, doc_lens, ques_lens, option_lens = batch

                            with torch.no_grad():
                                if args.NEW_MODEL:
                                    outputs = model(input_ids, input_mask, segment_ids, label_ids)
                                    tmp_eval_loss = outputs[0]
                                    outputs = model(input_ids, input_mask, segment_ids)
                                    logits = outputs[0]
                                else:
                                    outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
                                    tmp_eval_loss = outputs.loss
                                    outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                                    logits = outputs.logits

                            logits = logits.detach().cpu().numpy()
                            label_ids = label_ids.to('cpu').numpy()
                            tmp_eval_accuracy = accuracy(logits, label_ids)

                            eval_loss += tmp_eval_loss.mean().item()
                            eval_accuracy += tmp_eval_accuracy

                            nb_eval_examples += input_ids.size(0)
                            nb_eval_steps += 1

                        eval_loss = eval_loss / nb_eval_steps
                        eval_accuracy = eval_accuracy / nb_eval_examples

                        result = {'dev_eval_loss': eval_loss,
                                  'dev_eval_accuracy': eval_accuracy,
                                  'global_step': global_step}

                        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                        with open(output_eval_file, "a+") as writer_eval:
                            logger.info("***** Dev results *****")
                            for key in sorted(result.keys()):
                                logger.info("  %s = %s", key, str(result[key]))
                                writer_eval.write("%s = %s\n" % (key, str(result[key])))

                if is_main_process():
                    train_iter.set_postfix(loss=loss.item())
                writer.add_scalar('loss', loss.item(), global_step=global_step)

                if step != 0 and global_step == step:
                    break

    finish_time = time.time()
    writer.close()
    # Save a trained model
    if is_main_process():
        logger.info("ete_time: {}, training_time: {}".format(finish_time-ete_start, finish_time-train_start))
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        output_model_config = os.path.join(args.output_dir, "bert_config.json")
        with open(output_model_config, "w") as cf:
            json.dump(model_to_save.config.to_dict(), cf)


if __name__ == "__main__":
    main()
