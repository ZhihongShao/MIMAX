import os
import json
import pickle as pkl
import tokenization
import collections
from tqdm import tqdm
import copy

import numpy as np
import nltk
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tokenization import BasicTokenizer
from transformers import BartTokenizer, BertTokenizer

from prepro_util import *

def get_dataloader(logger, args, input_file, is_training, \
                   batch_size, num_epochs, tokenizer, index=None):
    '''
    train_features: list of list of features, each feature is an object, each inner list of features is the block of features of strided doc spans of a paragraph
    prior_features: list of list of features, each feature is an object, each inner list of features is the block of features of A strided doc span
    '''

    n_paragraphs = args.n_paragraphs

    if (not is_training) and ',' in n_paragraphs:
        n_paragraphs = n_paragraphs.split(',')[-1]

    feature_save_path = input_file.replace('.json', '-{}-{}-{}.pkl'.format(
            args.max_seq_length, n_paragraphs, args.max_n_answers))

    if False and os.path.exists(feature_save_path): # TODO
        logger.info("Loading saved features from {}".format(feature_save_path))
        with open(feature_save_path, 'rb') as f:
            features = pkl.load(f)
            train_features = features['features']
            examples = features.get('examples', None)
    else:

        examples = read_squad_examples(
            logger=logger, args=args, input_file=input_file, debug=args.debug)

        train_features = convert_examples_to_features(
            logger=logger,
            args=args,
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            max_n_answers=args.max_n_answers if is_training else 1,
            is_training=is_training)
        if is_training:
            prior_features = convert_train_features_to_prior_features(
                logger=logger,
                args=args,
                examples=examples,
                train_tokenizer=tokenizer,
                train_features=train_features
            )
        else:
            prior_features = None
        if not args.debug:
            logger.info("Saving features to: {}".format(feature_save_path))
            save_features = {'features': train_features, 'prior_features': prior_features, 'examples': examples}
            with open(feature_save_path, 'wb') as f:
                pkl.dump(save_features, f)

    n_features = sum([len(f) for f in train_features])
    num_train_steps = int(len(train_features) / batch_size * num_epochs)

    if examples is not None:
        logger.info("  Num orig examples = %d", len(examples))
    logger.info("  Num split examples = %d", n_features)
    logger.info("  Batch size = %d", batch_size)
    if is_training:
        logger.info("  Num steps = %d", num_train_steps)

def convert_train_features_to_prior_features(logger, args, examples, train_tokenizer, train_features):
    prior_tokenizer = BartTokenizer.from_pretrained('/home/shaozhihong/.cache/torch/bart-base')
    prior_tokenizer.add_tokens(['<passage_span>', '[prog]'])
    train_tokenizer = BertTokenizer.from_pretrained('/home/shaozhihong/.cache/torch/bert-base-uncased')
    prior_features = []
    if args.verbose:
        train_features = tqdm(train_features, total=len(train_features), desc='Trained_feats->Prior_feats')
    for block in train_features:
        for feature in block:
            current_features = []
            train_tokens = feature.tokens
            qlen = train_tokens.index('[SEP]')
            for start_pos, end_pos, switch, mask in zip(feature.start_position, feature.end_position, feature.switch, feature.answer_mask):
                if mask == 0:
                    continue
                if switch == 0:
                    prog_tokens = prior_tokenizer.tokenize(' passage span (')
                    prog_attn_range = [None] * len(prog_tokens)
                    prog_tokens.extend(['<passage_span>', ')'])
                    prog_attn_range.extend([[start_pos - qlen - 1, end_pos - qlen - 1], None])
                elif switch == 1:
                    prog_tokens = prior_tokenizer.tokenize(' yes')
                    prog_attn_range = [None] * len(prog_tokens)
                elif switch == 2:
                    prog_tokens = prior_tokenizer.tokenize(' no')
                    prog_attn_range = [None] * len(prog_tokens)
                else:
                    prog_tokens = prior_tokenizer.tokenize(' no answer')
                    prog_attn_range = [None] * len(prog_tokens)
                qptokens = copy.deepcopy(train_tokens)
                for idx, tok in enumerate(qptokens):
                    if tok == train_tokenizer.sep_token:
                        qptokens[idx] = prior_tokenizer.sep_token
                qtokens = qptokens[1: qlen]
                ptokens = qptokens[qlen + 1: -1]
                indices = set()
                for inv in prog_attn_range:
                    if inv is not None:
                        indices.add(inv[0])
                        indices.add(inv[1] + 1)
                indices.add(0)
                indices.add(len(ptokens))
                indices = sorted(list(indices))
                input_tokens = []
                old_idx2new_idx = {}
                for st, ed in zip(indices[:-1], indices[1:]):
                    old_idx2new_idx[st] = len(input_tokens)
                    tokens = ptokens[st: ed]
                    input_tokens.extend(prior_tokenizer.tokenize(train_tokenizer.convert_tokens_to_string(tokens)))
                old_idx2new_idx[len(ptokens)] = len(input_tokens)
                input_tokens = ['<s>'] + input_tokens + ['[prog]'] + prog_tokens + ['</s>']
                input_ids = prior_tokenizer.convert_tokens_to_ids(input_tokens)
                decoder_input_ids = prior_tokenizer.encode(train_tokenizer.convert_tokens_to_string(qtokens))

                plen = input_tokens.index('[prog]')
                attn_map = [[1.0] * len(input_ids)] + \
                    [[1.0] * plen + [0.0] * (len(prog_tokens) + 1) + [1.0]] * (plen - 1) + \
                    [[1.0] + [0.0] * (plen - 1) + [1.0] * (len(prog_tokens) + 2) for _ in range(len(prog_tokens) + 1)] + \
                    [[1.0] * len(input_ids)]
                for idx, inv in enumerate(prog_attn_range):
                    if inv is None:
                        continue
                    inv = [old_idx2new_idx[inv[0]], old_idx2new_idx[inv[1] + 1]]
                    tmp = [0.0] * len(input_ids)
                    for i in range(inv[0] + 1, inv[1] + 1):
                        tmp[i] = 1.0
                    attn_map[plen + 1 + idx] = tmp
                current_features.append(PriorInputFeatures(
                    unique_id=feature.unique_id,
                    example_index=feature.example_index,
                    paragraph_index=feature.paragraph_index,
                    doc_span_index=feature.doc_span_index,
                    input_ids=input_ids,
                    attention_mask=attn_map,
                    decoder_input_ids=decoder_input_ids
                ))
            prior_features.append(current_features)
    return prior_features

def read_squad_examples(logger, args, input_file, debug):
    def _process_sent(sent):
        if type(sent) != str:
            return [_process_sent(s) for s in sent]
        return sent.replace('–', '-').replace('&', 'and').replace('&amp;', 'and')

    input_data = []
    for _input_file in input_file.split(','):
        logger.info("Loading {}".format(_input_file))
        with open(_input_file, "r") as f:
            this_data = [json.loads(line) for line in f.readlines()]
            if debug:
                this_data = this_data[:50]
            input_data += this_data

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    if args.verbose:
        input_data = tqdm(input_data)
    for entry in input_data:

        # doc_tokens_list1, char_to_word_offset_list = [], []

        # for tokens in entry['context']:
        #     paragraph_text = ' '.join(tokens)
        #     doc_tokens = []
        #     char_to_word_offset = []
        #     prev_is_whitespace = True
        #     for c in paragraph_text:
        #         if is_whitespace(c):
        #             prev_is_whitespace = True
        #         else:
        #             if prev_is_whitespace:
        #                 doc_tokens.append(c)
        #             else:
        #                 doc_tokens[-1] += c
        #             prev_is_whitespace = False
        #         char_to_word_offset.append(len(doc_tokens) - 1)
        #     doc_tokens_list1.append(doc_tokens)
        #     char_to_word_offset_list.append(char_to_word_offset)

        question_text = entry["question"]

        original_answers_list, start_positions_list, end_positions_list, switches_list = [], [], [], []

        # for (char_to_word_offset, answers) in zip(char_to_word_offset_list, entry['answers']):
        for answers in entry['answers']:
            if len(answers)==0:
                original_answers, start_positions, end_positions, switches = [""], [0], [0], [3]
            else:
                original_answers, start_positions, end_positions = [[a[key] for a in answers] for key in ['text', 'word_start', 'word_end']]
                switches = [0 for _ in answers]
            original_answers_list.append(original_answers)
            start_positions_list.append(start_positions)
            end_positions_list.append(end_positions)
            switches_list.append(switches)

        examples.append(SquadExample(
                qas_id=entry['id'],
                question_text=entry['question'],
                doc_tokens=entry['context'],
                paragraph_indices=list(range(len(entry['context']))),
                orig_answer_text=original_answers_list,
                all_answers=entry['final_answers'],
                start_position=start_positions_list,
                end_position=end_positions_list,
                switch=switches_list))
    if "test" in input_file:
        return examples
    n_answers = []
    for example in examples:
        has_answer=False
        for switches in example.switch:
            assert 0 in switches or switches==[3]
            if 0 in switches:
                n_answers.append(len(switches))
    logger.info("# answers  = %.1f %.1f %.1f %.1f" %(
        np.mean(n_answers), np.median(n_answers),
        np.percentile(n_answers, 95), np.percentile(n_answers, 99)))
    return examples

def convert_examples_to_features(logger, args, examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, max_n_answers, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 0

    truncated = []
    features = []
    features_with_truncated_answers = []

    if args.verbose:
        examples = tqdm(enumerate(examples))
    else:
        examples = enumerate(examples)

    for (example_index, example) in examples:
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        assert len(example.doc_tokens) == len(example.orig_answer_text) == \
            len(example.start_position) == len(example.end_position) == len(example.switch)

        current_features =  []

        for (paragraph_index, doc_tokens, original_answer_text_list, start_position_list, end_position_list, switch_list) in \
                zip(example.paragraph_indices, example.doc_tokens, example.orig_answer_text, example.start_position, \
                    example.end_position, example.switch):
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize([token], basic_done=True)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)
            tok_start_positions, tok_end_positions = [], []

            if is_training:
                for (orig_answer_text, start_position, end_position) in zip( \
                            original_answer_text_list, start_position_list, end_position_list):
                    tok_start_position = orig_to_tok_index[start_position]
                    if end_position < len(doc_tokens) - 1:
                        tok_end_position = orig_to_tok_index[end_position + 1] - 1
                    else:
                        tok_end_position = len(all_doc_tokens) - 1
                    (tok_start_position, tok_end_position) = _improve_answer_span(
                        all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                        orig_answer_text)
                    tok_start_positions.append(tok_start_position)
                    tok_end_positions.append(tok_end_position)
                to_be_same = [len(original_answer_text_list), \
                                    len(start_position_list), len(end_position_list),
                                    len(switch_list), \
                                    len(tok_start_positions), len(tok_end_positions)]
                assert all([x==to_be_same[0] for x in to_be_same])


            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)

            truncated.append(len(doc_spans))
            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                        split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                start_positions = []
                end_positions = []
                switches = []
                answer_mask = []
                if is_training:
                    for (orig_answer_text, start_position, end_position, switch, \
                                tok_start_position, tok_end_position) in zip(\
                                original_answer_text_list, start_position_list, end_position_list, \
                                switch_list, tok_start_positions, tok_end_positions):
                        if orig_answer_text not in ['yes', 'no'] or switch == 3:
                            # For training, if our document chunk does not contain an annotation
                            # we throw it out, since there is nothing to predict.
                            doc_start = doc_span.start
                            doc_end = doc_span.start + doc_span.length - 1
                            if (tok_start_position < doc_start or
                                    tok_end_position < doc_start or
                                    tok_start_position > doc_end or tok_end_position > doc_end):
                                continue
                            doc_offset = len(query_tokens) + 2
                            start_position = tok_start_position - doc_start + doc_offset
                            end_position = tok_end_position - doc_start + doc_offset
                        else:
                            start_position, end_position = 0, 0
                        start_positions.append(start_position)
                        end_positions.append(end_position)
                        switches.append(switch)
                    to_be_same = [len(start_positions), len(end_positions), len(switches)]
                    assert all([x==to_be_same[0] for x in to_be_same])

                    if sum(to_be_same) == 0:
                        start_positions = [0]
                        end_positions = [0]
                        switches = [3]

                    if len(start_positions) > max_n_answers:
                        features_with_truncated_answers.append(len(features))
                        start_positions = start_positions[:max_n_answers]
                        end_positions = end_positions[:max_n_answers]
                        switches = switches[:max_n_answers]
                    answer_mask = [1 for _ in range(len(start_positions))]
                    for _ in range(max_n_answers-len(start_positions)):
                        start_positions.append(0)
                        end_positions.append(0)
                        switches.append(0)
                        answer_mask.append(0)

                current_features.append(
                    InputFeatures(
                        unique_id=unique_id,
                        example_index=example_index,
                        paragraph_index=paragraph_index,
                        doc_span_index=doc_span_index,
                        doc_tokens=doc_tokens,
                        tokens=tokens,
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        start_position=start_positions,
                        end_position=end_positions,
                        switch=switches,
                        answer_mask=answer_mask))
                unique_id += 1
        features.append(current_features)

    logger.info("# of features per paragraph: %.1f"%(np.mean(truncated)))
    return features

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

