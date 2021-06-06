import os
import shutil
import json
import re
import time
import traceback
from tqdm import tqdm
import pickle

import numpy as np
import collections

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, DistributedSampler

import evaluate_qa

from Exceptions import *

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "switch"])

class CustomDatasetConfig:
    def __init__(self):
        self.cache_dir = './.cache'
        self.batch_size = 32
        self.clamp_log_prob_min = -300

class CustomDataset:

    TRAIN_MODE = 'train'
    EVAL_MODE = 'eval'
    PREDICT_MODE = 'predict'

    def __init__(self, predict_file, library, mode, local_rank, num_shards=1, config=CustomDatasetConfig()):
        self.mode = mode
        self.local_rank = local_rank
        self.predict_file = predict_file.split(",")
        self.library = library
        self.num_shards = num_shards
        self.epoch = 0
        self._shard_id = -1
        self.config = config
        if self.local_rank == 0:
            if os.path.exists(self.config.cache_dir):
                shutil.rmtree(self.config.cache_dir)
            os.makedirs(self.config.cache_dir, exist_ok=True)
        self.examples = None # original examples
        self.positive_guids = None # positive unique ids
        self.negative_guids = None # negative unique ids
        self.guid2train_feature = None # unique_id to feature of a doc span, list of InputFeature
        self.guid2prior_features = None # unique_id to features of a doc span, list of list of PriorInputFeature
        self.guid2programs = {}
        self.indices = None # indices inside data loader
        self.guids = None # current batch of unique_ids
        self._read_examples() # set the above unset variables

    def increment_epoch(self):
        self.epoch += 1
        self._shard_id = -1
        self._read_examples()

    def get_next_batch(self):
        try:
            batch = next(self._shard_dataloader_iter)
            return batch
        except:
            raise EndOfShardError()

    def get_num_train_features(self):
        return self.num_train_features

    def _read_examples(self):
        if self.epoch == 0 or len(self.predict_file) > 1:
            predict_file = self.predict_file[self.epoch % len(self.predict_file)]
            data = pickle.load(open(predict_file, "rb"))
            self.examples = data['examples']
            self.guid2prior_features = data['prior_features']
            self.positive_guids = []
            self.negative_guids = []
            self.guid2train_feature = []
            self.num_train_features = len(data['features'])
            for block in data['features']:
                for feature in block:
                    guid = feature.unique_id
                    if self.mode == self.TRAIN_MODE:
                        switch = []
                        programs = []
                        for st, ed, s, m in zip(feature.start_position, feature.end_position, feature.switch, feature.answer_mask):
                            if m == 0:
                                break
                            switch.append(s)
                            programs.append({
                                'start_position': st,
                                'end_position': ed,
                                'switch': s
                            })
                        if not switch:
                            continue
                        self.guid2train_feature.append(feature)
                        if 3 in switch:
                            self.negative_guids.append(guid)
                        else:
                            self.positive_guids.append(guid)
                        self.guid2programs[guid] = programs
                    else:
                        self.guid2train_feature.append(feature)
                        self.positive_guids.append(guid)
        if self.mode == self.TRAIN_MODE:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            shuffle_indices = torch.randperm(len(self.negative_guids), generator=g).tolist()
            self.negative_guids = [self.negative_guids[i] for i in shuffle_indices]
            self.indices = list(range(2 * len(self.positive_guids)))
        else:
            self.indices = list(range(len(self.positive_guids)))

    def __len__(self):
        return len(self.indices)

    def set_current_batch_guids(self, indices):
        if not isinstance(indices, (list, tuple)):
            indices = indices.cpu().numpy().tolist()
        self.guids = []
        for idx in indices:
            if self.mode == self.TRAIN_MODE:
                if idx % 2 == 0:
                    self.guids.append(self.positive_guids[int(idx / 2)])
                else:
                    self.guids.append(self.negative_guids[int(idx / 2) % len(self.negative_guids)])
            else:
                self.guids.append(self.positive_guids[idx])

    def _set_dataloader(self, indices, desc='Iteration'):
        dataset = TensorDataset(torch.LongTensor(indices))
        if self.local_rank != -1 and self.mode == self.TRAIN_MODE:
            self.sampler = DistributedSampler(dataset)
            self.sampler.set_epoch(self.epoch)
        else:
            self.sampler = RandomSampler(dataset) if self.mode == self.TRAIN_MODE else SequentialSampler(dataset)
        self._shard_dataloader = tqdm(DataLoader(dataset, batch_size=self.config.batch_size, sampler=self.sampler), desc=desc, disable=self.local_rank not in [-1, 0])
        self._shard_dataloader_iter = iter(self._shard_dataloader)

    def save(self, ckpt_dir, filename):
        json.dump(self.guid2programs, open(os.path.join(ckpt_dir, filename), "w"))

    def get_guid2programs(self):
        return self.guid2programs

    def prepare_next_shard(self, posterior_net, prior_net, dynamic_preparation=True, for_posterior=True, use_other_net_for_selection=True):
        if self.epoch != 0 and self._shard_id != -1:
            self.synchronize()
        if self._shard_id + 1 == self.num_shards:
            raise EndOfEpochError()
        self._shard_id += 1
        if self._shard_id == 0:
            if self.mode == self.TRAIN_MODE:
                g = torch.Generator()
                g.manual_seed(self.epoch)
                rand_indices = torch.randperm(len(self.indices), generator=g).tolist()
                self.indices = sorted(self.indices)
                self.indices = [self.indices[idx] for idx in rand_indices]
            num_samples_per_shard = (len(self.indices) + self.num_shards - 1) // self.num_shards
            self._indices_shards = []
            for _shard_id in range(self.num_shards):
                self._indices_shards.append(self.indices[_shard_id * num_samples_per_shard: (_shard_id + 1) * num_samples_per_shard])
        shard_indices = self._indices_shards[self._shard_id]
        if self.mode == self.TRAIN_MODE:
            self._set_dataloader(shard_indices)
            if not dynamic_preparation:
                self._set_dataloader(shard_indices, desc='Preparation')
                if posterior_net is not None:
                    posterior_net.eval()
                if prior_net is not None:
                    prior_net.eval()
                for batch_indices in self._shard_dataloader:
                    batch_indices = batch_indices[0]
                    self.set_current_batch_guids(batch_indices)
                    try:
                        if (for_posterior and use_other_net_for_selection) or (not for_posterior and not use_other_net_for_selection):
                            score_net = prior_net
                        else:
                            score_net = posterior_net
                        self.update_programs(score_net, compute_log_prob=True)
                    except Exception as err:
                        print("ERROR FOUND DURING UPDATING PROGRAMS", flush=True)
                        print(traceback.format_exc(), flush=True)
            self.synchronize()
            indices = self._get_trainable_indices(shard_indices)
            self.guid2prior_selection = {}
            self.guid2posterior_selection = {}
            if not dynamic_preparation:
                self._set_dataloader(indices, desc='sampling')
                for batch_indices in self._shard_dataloader:
                    batch_indices = batch_indices[0]
                    self.set_current_batch_guids(batch_indices)
                    self.sample_program(for_posterior, use_other_net_for_selection)
            self._set_dataloader(indices)
        else:
            self._set_dataloader(shard_indices)

    def _get_trainable_indices(self, indices):
        raise NotImplementedError()

    def get_best_program_results(self, logger, n_best_size, do_lower_case=True, verbose_logging=True, n_paragraphs=None):
        '''
        *   Arguments:
            *   criterion (str): 'answer' or 'gen_prob'
        '''
        all_results = []
        print(type(self.positive_guids[0]), flush=True)
        print(type(list(self.guid2programs.keys())[0]), flush=True)
        for guid in self.positive_guids:
            programs = self.guid2programs[guid]
            all_results.append(RawResult(
                unique_id=guid,
                start_logits=programs['start_logits'],
                end_logits=programs['end_logits'],
                switch=programs['switch']
            ))
        metrics, all_predictions, all_nbest_json = evaluate_qa.write_predictions_multi_processing(logger, self.examples, self.guid2train_feature, all_results, n_best_size, do_lower_case, verbose_logging, False, n_paragraphs)
        return metrics, all_predictions, all_nbest_json

    def synchronize(self):
        filename = 'tmp_{}_{}.cache'.format(self.epoch, self._shard_id)
        if self.local_rank == 0:
            for fname in os.listdir(self.config.cache_dir):
                patt = re.search(r'^tmp_(?P<epoch>-?\d+)_(?P<shard_id>-?\d+).cache$', fname)
                if patt is None:
                    continue
                if int(patt['epoch']) <= self.epoch or int(patt['shard_id']) <= self._shard_id:
                    os.remove(os.path.join(self.config.cache_dir, fname))
        if self.local_rank != -1 and dist.get_world_size() > 1:
            local_filename = os.path.join(self.config.cache_dir, filename + str(self.local_rank))
            local_shard_guid2programs = {}
            assert hasattr(self, '_shard_dataloader')
            for batch_indices in self._shard_dataloader:
                batch_indices = batch_indices[0]
                self.set_current_batch_guids(batch_indices)
                for guid in self.guids:
                    if guid in self.guid2programs:
                        local_shard_guid2programs[guid] = self.guid2programs[guid]
            json.dump(local_shard_guid2programs, open(local_filename, "w"))
            filenames = [os.path.join(self.config.cache_dir, filename + str(local_rank)) for local_rank in range(dist.get_world_size())]
            while not all(os.path.exists(fname) and time.time() - os.path.getmtime(fname) > 2 for fname in filenames):
                pass
            filename = os.path.join(self.config.cache_dir, filename)
            if self.local_rank == 0:
                print("Synchronizing data:", self.local_rank, flush=True)
                for fname in filenames:
                    if fname != local_filename:
                        data = {int(guid): programs for guid, programs in json.load(open(fname, "r", encoding='utf-8')).items()}
                        self.guid2programs.update(data)
                for fname in filenames:
                    os.remove(fname)
                json.dump(self.guid2programs, open(filename, "w", encoding='utf-8'))
                num_replicas = dist.get_world_size()
                while not all(os.path.exists(os.path.join(self.config.cache_dir, 'rank_{}_sync_{}_{}_ok'.format(local_rank + 1, self.epoch, self._shard_id))) for local_rank in range(num_replicas - 1)):
                    pass
                for local_rank in range(num_replicas - 1):
                    os.remove(os.path.join(self.config.cache_dir, "rank_{}_sync_{}_{}_ok".format(local_rank + 1, self.epoch, self._shard_id)))
            else:
                print("Waiting:", self.local_rank, flush=True)
                while not (os.path.exists(filename) and time.time() - os.path.getmtime(filename) > 2):
                    pass
                self.guid2programs = {int(guid): programs for guid, programs in json.load(open(filename, "r", encoding='utf-8')).items()}
                os.system("touch {}/rank_{}_sync_{}_{}_ok".format(self.config.cache_dir, self.local_rank, self.epoch, self._shard_id))
            print("Exit sync:", self.local_rank, flush=True)

    def update_programs(self, model, guids=None, start_logits=None, end_logits=None, switch_logits=None, compute_log_prob=True):
        '''
        *   Arguments:
            *   program_samples (torch.LongTensor): [batch, num_program_samples, max_program_len]
        *   Returns:
            *   avg_f1 (float)
            *   avg_em (float)
            *   avg_log_prob (float)
            *   cnt_of_inst_w_program (int)
        '''
        model.eval()
        if guids is None:
            guids = self.guids
        if hasattr(model, 'module'):
            is_prior = model.module.is_prior
        else:
            is_prior = model.is_prior

        batch_id2num_program = []

        update_new_candidates = (start_logits is not None and end_logits is not None and switch_logits is not None)
        if update_new_candidates:
            start_logits = start_logits.cpu().numpy().tolist()
            end_logits = end_logits.cpu().numpy().tolist()
            switch_logits = switch_logits.cpu().numpy().tolist()
        if self.mode == self.TRAIN_MODE:
            assert not update_new_candidates
            assert compute_log_prob
        else:
            assert update_new_candidates
            assert not compute_log_prob

        inp_guids = []
        for batch_id, guid in enumerate(guids):
            if update_new_candidates:
                self.guid2programs[guid] = {
                    'start_logits': start_logits[batch_id],
                    'end_logits': end_logits[batch_id],
                    'switch': switch_logits[batch_id]
                }

            if compute_log_prob:
                num_programs = len(self.guid2programs.get(guid, []))
                batch_id2num_program.append(num_programs)
                if num_programs > 0:
                    inp_guids.append(guid)

        # compute log_prob of generating the question given the passage & program
        log_prob_key = 'prior_log_prob' if is_prior else 'post_log_prob'
        if sum(batch_id2num_program) > 0 and compute_log_prob:
            if is_prior:
                inp_features = []
                for guid in inp_guids:
                    inp_features.extend(self.guid2prior_features[guid])
                inps = self.convert_features_to_inputs_for_Prior(inp_features)
            else:
                inps = self.convert_features_to_inputs_for_Posterior([self.guid2train_feature[guid] for guid in inp_guids])

            with torch.no_grad():
                log_probs = []
                num_inps = len(inps[list(inps.keys())[0]])
                for idx in range(0, num_inps, self.config.batch_size):
                    res = model(**{key: value[idx: idx + self.config.batch_size] for key, value in inps.items()})
                    log_probs.extend(res['log_prob'].cpu().numpy().tolist() if is_prior else res['batch_id2program_log_probs'].cpu().numpy().tolist())
            log_probs_iter = iter(log_probs)
            for batch_id, (start_idx, cnt) in enumerate(list(zip(np.cumsum([0] + batch_id2num_program), batch_id2num_program))):
                guid = guids[batch_id]
                if cnt == 0:
                    continue
                if is_prior:
                    for program in self.guid2programs[guid]:
                        program[log_prob_key] = next(log_probs_iter)
                else:
                    for program, log_prob in zip(self.guid2programs[guid], next(log_probs_iter)):
                        program[log_prob_key] = log_prob

    def _random_sample(self, programs, prob_key='post_log_prob'):
        log_probs = np.array([max(item[prob_key], self.config.clamp_log_prob_min) for item in programs])
        probs = np.exp(log_probs)
        probs = probs / np.sum(probs)
        idx = np.random.choice(list(range(len(probs))), size=1, p=probs)[0]
        return idx

    def _greedy_sample(self, programs, prob_key='prior_log_prob'):
        best_program = None
        best_idx = None
        for idx, program in enumerate(programs):
            if best_program is None or program[prob_key] > best_program[prob_key]:
                best_program = program
                best_idx = idx
        return best_idx

    def _sample_program(self, sample_type_for_prior, sample_type_for_posterior, use_other_net_for_selection):
        '''
        *   Arguments:
            *   sample_type_for_prior (str): 'random' or 'greedy'
            *   sample_type_for_posterior (str): 'random' or 'greedy'
        '''
        for batch_id in range(len(self.guids)):
            guid = self.guids[batch_id]
            if self.guid2programs.get(guid, []):
                programs = self.guid2programs[guid]
                if sample_type_for_prior is not None:
                    prob_key = 'post_log_prob' if use_other_net_for_selection else 'prior_log_prob'
                    sample_for_prior = self._random_sample(programs, prob_key=prob_key) if sample_type_for_prior == 'random' else self._greedy_sample(programs, prob_key=prob_key)
                    self.guid2prior_selection[guid] = sample_for_prior
                if sample_type_for_posterior is not None:
                    prob_key = 'prior_log_prob' if use_other_net_for_selection else 'post_log_prob'
                    sample_for_posterior = self._random_sample(programs, prob_key=prob_key) if sample_type_for_posterior == 'random' else self._greedy_sample(programs, prob_key=prob_key)
                    self.guid2posterior_selection[guid] = sample_for_posterior

    def sample_program(self, for_posterior, use_other_net_for_selection):
        raise NotImplementedError()

    def get_posterior_network_inputs(self, with_sample):
        inps = self.convert_features_to_inputs_for_Posterior([self.guid2train_feature[guid] for guid in self.guids])
        if with_sample:
            inps['loss_selection'] = torch.tensor([self.guid2posterior_selection[guid] for guid in self.guids])
        return inps

    def get_prior_network_inputs(self, with_sample):
        assert with_sample
        features = []
        for guid in self.guids:
            features.append(self.guid2prior_features[guid][self.guid2prior_selection[guid]])
        return self.convert_features_to_inputs_for_Prior(features)

    def convert_features_to_inputs_for_Prior(self, features):
        input_ids = []
        attention_mask = []
        decoder_input_ids = []

        for feature in features:
            input_ids.append(torch.tensor(feature.input_ids, dtype=torch.long))
            attention_mask.append(torch.tensor(feature.attention_mask, dtype=torch.float32))
            decoder_input_ids.append(torch.tensor(feature.decoder_input_ids, dtype=torch.long))
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.library.get_pad_token_id(is_prior=True))
        max_seq_len = max(len(attn_mask) for attn_mask in attention_mask)
        _attention_mask = torch.zeros([len(attention_mask), max_seq_len, max_seq_len], dtype=torch.float32)
        for batch_id, attn_mask in enumerate(attention_mask):
            _attention_mask[batch_id, :len(attn_mask), :len(attn_mask)] = attn_mask
        attention_mask = _attention_mask
        decoder_input_ids = pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.library.get_pad_token_id(is_prior=True))
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids
        }

    def convert_features_to_inputs_for_Posterior(self, features):
        is_train = bool(self.mode == self.TRAIN_MODE)
        input_ids = []
        attention_mask = []
        token_type_ids = []
        start_positions = [] if is_train else None
        end_positions = [] if is_train else None
        switch = [] if is_train else None
        answer_mask = [] if is_train else None
        for feature in features:
            input_ids.append(feature.input_ids)
            attention_mask.append(feature.input_mask)
            token_type_ids.append(feature.segment_ids)
            if is_train:
                start_positions.append(feature.start_position)
                end_positions.append(feature.end_position)
                switch.append(feature.switch)
                answer_mask.append(feature.answer_mask)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float32),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'start_positions': torch.tensor(start_positions, dtype=torch.long) if is_train else None,
            'end_positions': torch.tensor(end_positions, dtype=torch.long) if is_train else None,
            'switch': torch.tensor(switch, dtype=torch.long) if is_train else None,
            'answer_mask': torch.tensor(answer_mask, dtype=torch.float32) if is_train else None
        }
