# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""Run BERT on Question Answering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import collections
import logging
import json
import math
import os
import random
import six
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import tokenization

from prepro import get_dataloader

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--bert_vocab_file", default="vocab.txt", type=str, \
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--log_dir", default="out", type=str, \
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--train_file", type=str, default="")
    parser.add_argument("--test_file", type=str, default="")
    parser.add_argument("--eval_file", type=str, default="")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=300, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument('--max_n_answers', type=int, default=20)
    parser.add_argument('--n_paragraphs', type=str, default='40')
    parser.add_argument('--verbose', action="store_true", default=False)
    parser.add_argument('--debug', action="store_true", default=False)

    args = parser.parse_args()

    if os.path.exists(args.log_dir) and os.listdir(args.log_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(args.log_dir, "log.txt")),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.bert_vocab_file, do_lower_case=args.do_lower_case)

    n_train_files = len(args.train_file.split(','))
    for fidx, filename in enumerate(args.train_file.split(",") + [args.test_file, args.eval_file]):
        get_dataloader(
            logger=logger, args=args,
            input_file=filename,
            is_training=bool(fidx < n_train_files),
            batch_size=32,
            num_epochs=1,
            tokenizer=tokenizer)

if __name__ == '__main__':
    main()
