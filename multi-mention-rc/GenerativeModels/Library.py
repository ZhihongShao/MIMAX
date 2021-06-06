import os
import re
import numpy as np
import copy

from transformers import BertTokenizer, BartTokenizer

# input types
INDEX_TYPE = 'index'
STRING_TYPE = 'string'
# inp & out type
KEY_VALUE_TYPE = 'key-value'
SPAN_TYPE = 'span'
SPANS_TYPE = 'spans'
NUM_TYPE = 'number'
VALUE_TYPE = 'value'
NUMS_TYPE = 'numbers'
ANY_TYPE = 'any'
# special types
BOS_TYPE = '[bos]'
PAD_TYPE = '[pad]'
EOS_TYPE = '[eos]'
EOF_TYPE = '<eof>'

class Operation:
    def __init__(self, op_name, for_question, for_passage, inp_args_type, out_type, inf_last_arg=False):
        self.name = op_name
        self.for_question = for_question
        self.for_passage = for_passage
        self.inp_args_type = inp_args_type
        self.out_type = out_type
        self.inf_last_arg = inf_last_arg
        self.arg_id2choices = {}
        self.arg_id2rm_choices = {}
        self.arg_name2limit = {}
        self.allow_zero_args = (len(inp_args_type) == 0)

    def set_token_id(self, token_id):
        self.token_id = token_id

    def set_restricted_tokens_for_arg(self, arg_id, choices):
        assert isinstance(choices, (set, list, tuple))
        assert arg_id < len(self.inp_args_type)
        self.arg_id2choices[arg_id] = list(set(choices))

    def set_removed_tokens_for_arg(self, arg_id, choices):
        assert isinstance(choices, (set, list, tuple))
        assert arg_id < len(self.inp_args_type)
        self.arg_id2rm_choices[arg_id] = list(set(choices))

    def set_limit_of_arg(self, arg_name, limit):
        self.arg_name2limit[arg_name] = limit

    def get_restricted_tokens_for_arg(self, arg_id):
        return self.arg_id2choices.get(arg_id, [])

    def get_removed_tokens_for_arg(self, arg_id):
        return self.arg_id2rm_choices.get(arg_id, [])

    def get_limit_of_arg(self, arg_name):
        return self.arg_name2limit.get(arg_name, np.inf)

class Constant:
    def __init__(self, const_name, for_question, for_passage, out_type):
        self.name = const_name
        self.for_question = for_question
        self.for_passage = for_passage
        self.out_type = out_type

    def set_token_id(self, token_id):
        self.token_id = token_id

class Library:
    def __init__(self, max_num_args, max_program_depth, prior_textual_tokens_dir, post_textual_tokens_dir):
        prior_tokenizer_class = BartTokenizer if 'bart' in prior_textual_tokens_dir.lower() else BertTokenizer
        post_tokenizer_class = BartTokenizer if 'bart' in post_textual_tokens_dir.lower() else BertTokenizer
        self.prior_tokenizer = prior_tokenizer_class.from_pretrained(prior_textual_tokens_dir)
        self.post_tokenizer = post_tokenizer_class.from_pretrained(post_textual_tokens_dir)

        self.max_num_args = max_num_args
        self.max_program_depth = max_program_depth

        self._operations = []
        self._constants = []

        self._is_finalized = False

        self.register_op('[bos]', True, True, [ANY_TYPE, EOS_TYPE], BOS_TYPE, inf_last_arg=False)
        self.register_op('[eos]', True, True, [PAD_TYPE], EOS_TYPE, inf_last_arg=True)
        self.register_op('[pad]', True, True, [PAD_TYPE], PAD_TYPE, inf_last_arg=True)
        self.register_op('<eof>', True, True, [], EOF_TYPE, inf_last_arg=False)

    def get_cls_token_id(self, is_prior=False):
        assert self._is_finalized
        tokenizer = self.prior_tokenizer if is_prior else self.post_tokenizer
        return tokenizer.cls_token_id

    def get_sep_token_id(self, is_prior=False):
        assert self._is_finalized
        tokenizer = self.prior_tokenizer if is_prior else self.post_tokenizer
        return tokenizer.sep_token_id

    def get_bos_token_id(self, is_prior=False):
        assert self._is_finalized
        tokenizer = self.prior_tokenizer if is_prior else self.post_tokenizer
        return tokenizer.bos_token_id

    def get_pad_token_id(self, is_prior=False):
        assert self._is_finalized
        tokenizer = self.prior_tokenizer if is_prior else self.post_tokenizer
        return tokenizer.pad_token_id

    def get_eof_token_id(self, is_prior=False):
        assert self._is_finalized
        tokenizer = self.prior_tokenizer if is_prior else self.post_tokenizer
        return tokenizer.eof_token_id

    def get_eos_token_id(self, is_prior=False):
        assert self._is_finalized
        tokenizer = self.prior_tokenizer if is_prior else self.post_tokenizer
        return tokenizer.eos_token_id

    def get_textual_op_lst(self, is_prior):
        assert self._is_finalized
        res = []
        for op in self._operations:
            if (len(op.inp_args_type) == 1 and not op.inf_last_arg and op.inp_args_type[0] == STRING_TYPE) or \
                (is_prior and self.is_question_op(op.token_id) and (self.is_span_index_op(op.token_id) or self.is_spans_select_op(op.token_id))):
                res.append(op.token_id)
        return res

    def get_num_reserved_consts(self):
        assert self._is_finalized
        return len(self._constants)

    def get_reserved_token_index(self, token):
        assert self._is_finalized
        for idx, op in enumerate(self._constants + self._operations):
            if op.name == token:
                return idx

    def get_new_num_encoding_tokens(self, is_prior=False):
        tokenizer = self.prior_tokenizer if is_prior else self.post_tokenizer
        return tokenizer.vocab_size + len(self.additional_tokens) + len(self.consts_name) + 4

    def get_num_textual_tokens(self, is_prior):
        assert self._is_finalized
        # question generation || filling generation
        if is_prior or self.get_textual_op_lst(is_prior=is_prior):
            tokenizer = self.prior_tokenizer if is_prior else self.post_tokenizer
            return tokenizer.vocab_size + len(self.additional_tokens) + len(self.consts_name) + 4
        else:
            return 0

    def get_num_reserved_tokens(self):
        assert self._is_finalized
        return len(self.reserved_tokens)

    # TODO: if we do library updating
    def get_num_tokens(self, is_prior):
        assert self._is_finalized
        # if is_prior or self.get_textual_op_lst(is_prior=is_prior):
        #     return len(self.post_tokenizer)
        # else:
        #     return self.get_num_reserved_tokens()
        if is_prior:
            return len(self.prior_tokenizer)
        else:
            return len(self.post_tokenizer)

    def get_reserved_token_id_offset(self, is_prior=False):
        assert self._is_finalized
        tokenizer = self.prior_tokenizer if is_prior else self.post_tokenizer
        return tokenizer.vocab_size + len(self.additional_tokens)

    def get_op2arg_budget(self, is_prior, is_in_sketch):
        op2arg_budget = (np.sum(self.get_op2args2mask(is_prior, is_in_sketch), 1) > 0.5).astype(np.float32) * self.max_num_args
        for op_id, op in enumerate(self._operations):
            for idx, token in enumerate(self._constants + self._operations):
                if op2arg_budget[op_id, idx] > 0.5:
                    limit = min(op.get_limit_of_arg(token.name), self.max_num_args)
                    op2arg_budget[op_id, idx] = limit
        return op2arg_budget

    def get_op2args2mask(self, is_prior, is_in_sketch, postprocess_hook=None):
        assert self._is_finalized
        op2args2mask = []
        num_reserved_consts = len(self._constants)
        num_ops = len(self._operations)
        for op in self._operations:
            if (is_in_sketch and self.is_filling_op(op.token_id)) or (not is_in_sketch and is_prior and self.is_question_op(op.token_id)):
                op2args2mask.append([[]])
            else:
                args2mask = []
                for_question = int(op.for_question)
                for_passage = int(op.for_passage)
                for arg_id, arg in enumerate(op.inp_args_type):
                    mask = []
                    for idx, token in enumerate(self._constants + self._operations):
                        _for_question = int(token.for_question)
                        _for_passage = int(token.for_passage)
                        if (not op.get_restricted_tokens_for_arg(arg_id) or token.name in op.get_restricted_tokens_for_arg(arg_id)) \
                            and (
                                not op.get_removed_tokens_for_arg(arg_id) or token.name not in op.get_removed_tokens_for_arg(arg_id)
                            ) \
                                and (
                                    for_question * _for_question + for_passage * _for_passage > 0 # TODO: check
                                    # for_question * _for_question + for_passage * _for_passage >= _for_question + _for_passage or \
                                    # (idx < num_reserved_consts and for_question * _for_question + for_passage * _for_passage >= for_question + for_passage)
                                ) \
                                    and (
                                        (arg == token.out_type and idx >= num_reserved_consts) \
                                        or (arg == ANY_TYPE and token.out_type not in [PAD_TYPE, EOS_TYPE, EOF_TYPE]) \
                                        or (op.out_type == NUM_TYPE and arg == INDEX_TYPE and idx < num_reserved_consts and token.out_type == NUM_TYPE) \
                                        # or (arg_id == len(op.inp_args_type) - 1 and op.inf_last_arg and ((arg == SPAN_TYPE and token.out_type == SPANS_TYPE) or (arg == NUM_TYPE and token.out_type == NUMS_TYPE)))
                                    ):
                            mask.append(1.0)
                        else:
                            mask.append(0.0)
                    if op.allow_zero_args:
                        mask[num_reserved_consts + 3] = 1.0 # <eof>
                    args2mask.append(mask)
                if op.token_id not in [self.get_bos_token_id(is_prior=is_prior), self.get_pad_token_id(is_prior=is_prior), self.get_eos_token_id(is_prior=is_prior), self.get_eof_token_id(is_prior=is_prior)]:
                    mask = copy.deepcopy(args2mask[-1]) if op.inf_last_arg else [0.0] * (num_reserved_consts + num_ops)
                    mask[num_reserved_consts + 3] = 1.0 # <eof>
                    args2mask.append(mask)
                op2args2mask.append(args2mask)
        max_arg_cnt = max(len(args2mask) for args2mask in op2args2mask)
        res = np.zeros((num_ops, max_arg_cnt, num_reserved_consts + num_ops), dtype=np.float32)
        for op_id, args2mask in enumerate(op2args2mask):
            if len(args2mask) < max_arg_cnt and self._operations[op_id].inf_last_arg:
                args2mask.extend([args2mask[-1]] * (max_arg_cnt - len(args2mask)))
            for arg_id, mask in enumerate(args2mask):
                if mask:
                    res[op_id, arg_id, :len(mask)] = mask
        res[:, :, num_reserved_consts] = 0.0 # can't generate [bos]
        res[0, :, :num_reserved_consts] = 0.0 # [bos] can't generate reserved constants
        if postprocess_hook is not None:
            res = postprocess_hook(res)
        print("----op2args2mask----")
        print("is_prior", is_prior)
        print("is_in_sketch", is_in_sketch)
        for j, op in enumerate(self._operations):
            print(op.name)
            for i, token in enumerate(self._constants + self._operations):
                print(token.name, res[j, 0, i], end="\t")
            print()
            for i, token in enumerate(self._constants + self._operations):
                print(token.name, res[j, 1, i], end="\t")
            print()
            for i, token in enumerate(self._constants + self._operations):
                print(token.name, res[j, 2, i], end="\t")
            print()
            print()
        print("------------", flush=True)
        return res

    def get_op2num_args(self, is_prior, is_in_sketch):
        assert self._is_finalized
        op_id2num = []
        for op in self._operations:
            if is_in_sketch and self.is_filling_op(op.token_id):
                op_id2num.append(0)
            elif not is_in_sketch and is_prior and self.is_question_op(op.token_id):
                op_id2num.append(1)
            elif op.token_id not in [self.get_bos_token_id(is_prior=is_prior), self.get_eof_token_id(is_prior=is_prior)]:
                op_id2num.append(len(op.inp_args_type) + 1 if not op.inf_last_arg else -1)
            else:
                op_id2num.append(len(op.inp_args_type))
        print("-----op2num_args-----")
        print("is_prior", is_prior)
        print("is_in_sketch", is_in_sketch)
        for num, op in zip(op_id2num, self._operations):
            print(op.name, "num_args", num)
        print()
        print("---------------------", flush=True)
        return np.array(op_id2num, dtype=np.int32)

    def get_op2args2inp_selection(self, is_prior, is_in_sketch, postprocess_hook=None):
        assert self._is_finalized
        op2args2inp_mask_selection = []
        op2args2inp_enc_selection = []
        for op in self._operations:
            if (is_in_sketch and self.is_filling_op(op.token_id)) or (not is_in_sketch and is_prior and self.is_question_op(op.token_id)):
                op2args2inp_mask_selection.append([])
                op2args2inp_enc_selection.append([])
            else:
                mask_tmp = []
                enc_tmp = []
                for_passage = int(not self.is_question_op(op.token_id))
                output_span = (op.out_type == SPAN_TYPE)
                output_num = int(op.out_type in [NUM_TYPE, VALUE_TYPE])
                output_value = int(op.out_type == VALUE_TYPE)
                for arg in op.inp_args_type:
                    if (output_span or output_num) and arg == INDEX_TYPE:
                        mask_tmp.append(for_passage * 2 + output_num)
                    else:
                        mask_tmp.append(-1)
                    if (output_span or output_value or output_num) and arg == INDEX_TYPE:
                        enc_tmp.append(for_passage * 2 + output_value)
                    else:
                        enc_tmp.append(-1)
                op2args2inp_mask_selection.append(mask_tmp + [-1])
                op2args2inp_enc_selection.append(enc_tmp + [-1])
        max_arg_cnt = max(len(line) for line in op2args2inp_mask_selection)
        op2args2inp_mask_selection = [line + [-1] * (max_arg_cnt - len(line)) for line in op2args2inp_mask_selection]
        op2args2inp_enc_selection = [line + [-1] * (max_arg_cnt - len(line)) for line in op2args2inp_enc_selection]
        if postprocess_hook is not None:
            op2args2inp_mask_selection, op2args2inp_enc_selection = postprocess_hook(np.array(op2args2inp_mask_selection, dtype=np.int32), np.array(op2args2inp_enc_selection, dtype=np.int32))
        for title, data in zip(['op2args2inp_mask_selection', 'op2args2inp_enc_selection'], [op2args2inp_mask_selection, op2args2inp_enc_selection]):
            print("----{}-----".format(title))
            print("is_prior", is_prior)
            print("is_in_sketch", is_in_sketch)
            for op_id, op in enumerate(self._operations):
                print(op.name)
                for arg_id in range(max_arg_cnt):
                    print("<arg:{}>".format(arg_id), data[op_id][arg_id], end="\t")
                print()
                print()
            print("---------------", flush=True)
        return op2args2inp_mask_selection, op2args2inp_enc_selection

    def is_filling_op(self, token_id):
        assert self._is_finalized
        for op in self._operations:
            if op.token_id == token_id:
                return self.is_index_op(token_id) or self.is_span_index_op(token_id) or self.is_spans_select_op(token_id)
        return False

    # question specific op
    def is_question_op(self, token_id):
        assert self._is_finalized
        for op in self._operations:
            if op.token_id == token_id:
                return op.for_question and not op.for_passage
        return False

    def is_index_op(self, token_id):
        assert self._is_finalized
        for op in self._operations:
            if op.token_id == token_id:
                return not op.inf_last_arg and len(op.inp_args_type) == 1 and op.inp_args_type[0] == INDEX_TYPE
        return False

    def is_span_index_op(self, token_id):
        assert self._is_finalized
        for op in self._operations:
            if op.token_id == token_id:
                return op.out_type == SPAN_TYPE and not op.inf_last_arg and len(op.inp_args_type) == 2 and op.inp_args_type[0] == INDEX_TYPE and op.inp_args_type[1] == INDEX_TYPE
        return False

    def is_spans_select_op(self, token_id):
        assert self._is_finalized
        for op in self._operations:
            if op.token_id == token_id and op.out_type == SPANS_TYPE and set(op.inp_args_type) == set([SPAN_TYPE]) and len(op.get_restricted_tokens_for_arg(0)) == 1:
                token = op.get_restricted_tokens_for_arg(0)[0]
                for op in self._operations:
                    if op.name == token and self.is_span_index_op(op.token_id):
                        return True
                break
        return False

    def is_op_with_args(self, token_id):
        assert self._is_finalized
        for op in self._operations:
            if op.token_id == token_id:
                return len(op.inp_args_type) > 0
        return False

    def is_index(self, token_id):
        assert self._is_finalized
        return token_id >= self.get_num_tokens(is_prior=False)

    def register_op(self, op_name, for_question, for_passage, inp_args_type, out_type, inf_last_arg=False):
        assert not self._is_finalized
        for op in self._operations:
            assert op.name != op_name
        self._operations.append(Operation(op_name, for_question, for_passage, inp_args_type, out_type, inf_last_arg))

    def register_const(self, const_name, for_question, for_passage, out_type):
        assert not self._is_finalized
        for const in self._constants:
            assert const.name != const_name
        self._constants.append(Constant(const_name, for_question, for_passage, out_type))

    def set_restricted_args_for_op(self, op_name, arg_id, choices):
        assert not self._is_finalized
        for op in self._operations:
            if op.name == op_name:
                op.set_restricted_tokens_for_arg(arg_id, choices)
                return
        raise RuntimeError("No such op as `{}`".format(op_name))

    def remove_args_for_op(self, op_name, arg_id, choices):
        assert not self._is_finalized
        for op in self._operations:
            if op.name == op_name:
                op.set_removed_tokens_for_arg(arg_id, choices)
                return
        raise RuntimeError("No such op as `{}`".format(op_name))

    def set_zero_args_allowance_to_op(self, op_name):
        assert not self._is_finalized
        for op in self._operations:
            if op.name == op_name:
                op.allow_zero_args = True
                break

    def set_limit_of_arg_for_op(self, op_name, arg_name, limit):
        assert not self._is_finalized
        for op in self._operations:
            if op.name == op_name:
                op.set_limit_of_arg(arg_name, limit)
                return
        raise RuntimeError("No such op as `{}`".format(op_name))

    def finalize(self, additional_tokens):
        assert not self._is_finalized
        self.consts_name = [const.name for const in self._constants]
        self.ops_name = [op.name for op in self._operations]
        self.reserved_tokens = self.consts_name + self.ops_name
        self.additional_tokens = additional_tokens
        for is_prior, tokenizer in zip([True, False], [self.prior_tokenizer, self.post_tokenizer]):
            tokenizer.add_tokens(additional_tokens)
            additional_tokens = additional_tokens
            tokenizer.add_tokens(self.reserved_tokens)
            if not is_prior:
                tokenizer.bos_token = '[bos]'
                tokenizer.eos_token = '[eos]'
                tokenizer.pad_token = '[pad]'
                tokenizer.eof_token = '<eof>'
                tokenizer.eof_token_id = tokenizer.encode('<eof>')[1]

        for token in self._operations + self._constants:
            token.set_token_id(self.post_tokenizer.encode(token.name)[1])

        self._is_finalized = True

        print("-------Op Categories-------")
        for op in self._operations:
            msg = op.name + ": "
            if self.is_filling_op(op.token_id):
                msg += "filling-op "
            if self.is_index_op(op.token_id):
                msg += "index-op "
            if self.is_question_op(op.token_id):
                msg += "question-op "
            if self.is_span_index_op(op.token_id):
                msg += "span-index-op "
            if self.is_spans_select_op(op.token_id):
                msg += "spans-select-op "
            print(msg, flush=True)
        print("-" * 14, flush=True)

    def convert_tokens_to_ids(self, tokens, add_eos_token_id=False, is_prior=False):
        tokenizer = self.prior_tokenizer if is_prior else self.post_tokenizer
        token_ids = []
        for token in tokens:
            res = re.match(r'^<index:(\d+)>$', token)
            if res is not None:
                token_ids.append(int(res.groups()[0]) + self.get_num_tokens(is_prior=False))
            else:
                token_ids.extend(tokenizer.convert_tokens_to_ids([token]))
        if add_eos_token_id:
            token_ids += [self.get_eos_token_id(is_prior=is_prior)]
        return token_ids

    def tokenize(self, text, is_prior=False):
        tokenizer = self.prior_tokenizer if is_prior else self.post_tokenizer
        return tokenizer.tokenize(text)

    def encode(self, text, add_bos_token_id=False, add_eos_token_id=False, is_prior=False):
        tokenizer = self.prior_tokenizer if is_prior else self.post_tokenizer
        token_ids = []
        for token in text.split():
            res = re.match(r'^<index:(\d+)>$', token)
            if res is not None:
                token_ids.append(int(res.groups()[0]) + self.get_num_tokens(is_prior=False))
            else:
                token_ids.extend(tokenizer.convert_tokens_to_ids(self.tokenize(token, is_prior=is_prior)))
        if add_bos_token_id:
            token_ids = [self.get_bos_token_id(is_prior=is_prior)] + token_ids
        if add_eos_token_id:
            token_ids += [self.get_eos_token_id(is_prior=is_prior)]
        return token_ids

    def decode(self, token_ids, remove_special_tokens=True, is_prior=False):
        tokenizer = self.prior_tokenizer if is_prior else self.post_tokenizer
        tokens = []
        for token_id in token_ids:
            if remove_special_tokens:
                if token_id in [self.get_bos_token_id(is_prior=is_prior), self.get_pad_token_id(is_prior=is_prior), self.get_eos_token_id(is_prior=is_prior)]:
                    continue
            if self.is_index(token_id):
                tokens.append('<index:{}>'.format(token_id - self.get_num_tokens(is_prior=False)))
            else:
                tokens.append(tokenizer.convert_ids_to_tokens([token_id])[0])
        return tokenizer.convert_tokens_to_string(tokens)

    def get_leaf_mask_for_sketch(self):
        raise NotImplementedError()

    def build(self):
        raise NotImplementedError()

    def execute(
        self,
        question_text,
        question_passage_tokens,
        question_token_indices,
        question_number_indices,
        question_numbers,
        passage_text,
        passage_token_indices,
        passage_number_indices,
        passage_numbers,
        program_token_ids,
        answers
    ):
        raise NotImplementedError()
