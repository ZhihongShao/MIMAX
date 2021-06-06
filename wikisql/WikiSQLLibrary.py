import re
import json
import numpy as np
import traceback
from itertools import chain

from GenerativeModels import Library

class WikiSQLLibrary(Library.Library):
    def __init__(self, max_num_args, max_program_depth, prior_textual_tokens_dir, post_textual_tokens_dir):
        super().__init__(max_num_args, max_program_depth, prior_textual_tokens_dir, post_textual_tokens_dir)
        self.build()

    def build(self):
        self.agg_ops = ['none<agg>', 'max<agg>', 'min<agg>', 'count<agg>', 'sum<agg>', 'avg<agg>']
        self.cond_ops = ['=<cond>', '><cond>', '<<cond>', 'OP<cond>']
        self.finalize(['[col]', '[prog]', '[tab]', 'select<clause>', 'where<clause>', 'and<clause>', '<question_span>', '<passage_span>'] + self.agg_ops + self.cond_ops)

        self.op2tokens = {
            'none<agg>': 'none',
            'max<agg>': 'max',
            'min<agg>': 'min',
            'count<agg>': 'count',
            'sum<agg>': 'sum',
            'avg<agg>': "average",

            '=<cond>': '=',
            '><cond>': '>',
            '<<cond>': '<',
            'OP<cond>': 'op',

            'select<clause>': 'select',
            'where<clause>': 'where',
            'and<clause>': 'and',
        }

    def get_num_tags(self):
        return 2

    def get_num_cond_ops(self):
        return len(self.cond_ops)
    
    def get_num_agg_ops(self):
        return len(self.agg_ops)

    def decode_program(self, sql):
        tokens = ['select<clause>', self.agg_ops[sql['agg']], '(', '<passage_span>', ')', 'where<clause>']
        spans = [['table', int(sql['sel']), 3]]
        for cond in sql['conds']:
            spans.append(['table', int(cond[0]), len(tokens)])
            tokens.extend(['<passage_span>', self.cond_ops[cond[1]]] + self.prior_tokenizer.tokenize(" " + cond[2].strip()) + ['and<clause>'])
        if sql['conds']:
            tokens = tokens[:-1]
        _tokens = []
        for token in tokens:
            if token in self.op2tokens:
                lst = self.prior_tokenizer.tokenize(" " + self.op2tokens[token])
                assert len(lst) == 1
                token = lst[0]
            _tokens.append(token)
        return _tokens, spans
