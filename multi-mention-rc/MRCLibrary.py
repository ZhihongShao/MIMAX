import os
import re
import json
import numpy as np
import traceback
from itertools import chain
from transformers import BertTokenizer, BartTokenizer
import tokenization

class MRCLibrary:
    def __init__(self, prior_textual_tokens_dir, post_textual_tokens_dir):
        prior_tokenizer_class = BartTokenizer if 'bart' in prior_textual_tokens_dir.lower() else BertTokenizer
        self.prior_tokenizer = prior_tokenizer_class.from_pretrained(prior_textual_tokens_dir)
        self.prior_tokenizer.add_tokens(['<passage_span>', '[prog]'])
        # self.post_tokenizer = tokenization.FullTokenizer(os.path.join(post_textual_tokens_dir, 'vocab.txt'), do_lower_case=True)
        self.post_tokenizer = BertTokenizer.from_pretrained(post_textual_tokens_dir)

    def get_pad_token_id(self, is_prior):
        assert is_prior
        return self.prior_tokenizer.pad_token_id
    
    def get_num_tokens(self, is_prior):
        assert is_prior
        return len(self.prior_tokenizer)
