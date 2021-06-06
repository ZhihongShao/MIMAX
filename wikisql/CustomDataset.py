import os
import shutil
import copy
import json
import re
import time
import traceback
from tqdm import tqdm

import numpy as np

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, DistributedSampler

from sqlnet.dbengine import DBEngine
from GenerativeModels.sqlova.utils import utils_wikisql

from Exceptions import *

class CustomDatasetConfig:
    def __init__(self):
        self.cache_dir = './.cache'
        self.batch_size = 32
        self.clamp_log_prob_min = -300
        self.exe_acc_threshold_for_sample = 0.5
        self.use_prior = True
        self.subsample = True

class CustomDataset:

    TRAIN_MODE = 'train'
    EVAL_MODE = 'eval'
    PREDICT_MODE = 'predict'

    def __init__(self, predict_file, table_file, db_file, library, mode, local_rank, num_shards=1, guid2programs_file=None, config=CustomDatasetConfig()):
        self.mode = mode
        self.local_rank = local_rank
        self.predict_file = predict_file
        self.table_file = table_file
        self.db_file = db_file
        self.library = library
        self.num_shards = num_shards
        self.epoch = 0
        self._shard_id = -1
        self.guid2programs_file = guid2programs_file
        self.config = config
        if self.local_rank == 0:
            if os.path.exists(self.config.cache_dir):
                shutil.rmtree(self.config.cache_dir)
            os.makedirs(self.config.cache_dir, exist_ok=True)
        self.guid_lst, self.guid2instance, self.guid2programs, self.id2table = self._read_examples(self.predict_file, self.table_file)
        self.engine = DBEngine(self.db_file)
        self.guid_lst = sorted(self.guid_lst)
        if self.mode == self.TRAIN_MODE and guid2programs_file is not None:
            self.guid2programs = self._read_guid2programs(guid2programs_file)
        if self.guid2programs:
            self.execute_augmented_programs()
        self.guids = None # current batch

    def increment_epoch(self):
        self.epoch += 1
        self._shard_id = -1

    def get_next_batch(self):
        try:
            batch = next(self._shard_dataloader_iter)
            return batch
        except:
            raise EndOfShardError()

    def prepare_next_shard(self, posterior_net, score_net, dynamic_preparation=True, for_posterior=True, use_other_net_for_selection=True):
        if self.epoch != 0 and self._shard_id != -1:
            # self.synchronize()
            pass
        if self._shard_id + 1 == self.num_shards:
            raise EndOfEpochError()
        self._shard_id += 1
        if self._shard_id == 0:
            if self.mode == self.TRAIN_MODE:
                g = torch.Generator()
                g.manual_seed(self.epoch)
                rand_indices = torch.randperm(len(self.guid_lst), generator=g).tolist()
                self.guid_lst = sorted(self.guid_lst)
                self.guid_lst = [self.guid_lst[idx] for idx in rand_indices]
            num_samples_per_shard = (len(self.guid_lst) + self.num_shards - 1) // self.num_shards
            self._guid_shards = []
            for _shard_id in range(self.num_shards):
                self._guid_shards.append(self.guid_lst[_shard_id * num_samples_per_shard: (_shard_id + 1) * num_samples_per_shard])
        shard_guids = self._guid_shards[self._shard_id]
        if self.mode == self.TRAIN_MODE:
            if not dynamic_preparation:
                self._set_guid_dataloader(shard_guids, desc='Preparation')
                if posterior_net is not None:
                    posterior_net.eval()
                if score_net is not None:
                    score_net.eval()
                metrics = {}
                num_batch = 0
                for guid_batch in self._shard_dataloader:
                    guid_batch = guid_batch[0]
                    self.set_current_batch_guids(guid_batch)
                    try:
                        batch_metrics = self.update_programs(posterior_net, score_net, compute_log_prob=True, subsample=self.config.subsample)
                        for key, val in batch_metrics.items():
                            metrics[key] = metrics.get(key, 0) + val
                        num_batch += 1
                    except Exception as err:
                        print("ERROR FOUND DURING UPDATING PROGRAMS", flush=True)
                        print(traceback.format_exc(), flush=True)
            self.synchronize()
            guids = self._get_trainable_guids(shard_guids)
            self.guid2prior_sample = {}
            self.guid2posterior_sample = {}
            if not dynamic_preparation:
                self._set_guid_dataloader(guids, desc='sampling')
                for guid_batch in self._shard_dataloader:
                    guid_batch = guid_batch[0]
                    self.set_current_batch_guids(guid_batch)
                    self.sample_program(for_posterior, use_other_net_for_selection)
                for key, val in metrics.items():
                    metrics[key] = val / num_batch
            self._set_guid_dataloader(guids)
            if not dynamic_preparation:
                return metrics
        else:
            self._set_guid_dataloader(shard_guids)

    def _get_trainable_guids(self, guids):
        raise NotImplementedError()

    def _set_guid_dataloader(self, guids, desc='Iteration'):
        dataset = TensorDataset(torch.LongTensor(guids))
        if self.local_rank != -1 and self.mode == self.TRAIN_MODE:
            self.sampler = DistributedSampler(dataset)
            self.sampler.set_epoch(self.epoch)
        else:
            self.sampler = RandomSampler(dataset) if self.mode == self.TRAIN_MODE else SequentialSampler(dataset)
        self._shard_dataloader = tqdm(DataLoader(dataset, batch_size=self.config.batch_size, sampler=self.sampler), desc=desc, disable=self.local_rank not in [-1, 0])
        self._shard_dataloader_iter = iter(self._shard_dataloader)

    def __len__(self):
        return len(self.guid_lst)

    def save(self, ckpt_dir, filename):
        guid2programs = {}
        for guid, programs in self.guid2programs.items():
            _programs = []
            for program in programs:
                _program = {}
                for key, val in program.items():
                    if not key.endswith('acc') and key != 'wvi_corenlp':
                        _program[key] = val
                _programs.append(_program)
            guid2programs[guid] = _programs
        json.dump(guid2programs, open(os.path.join(ckpt_dir, filename), "w"))

    def _read_examples(self, predict_file, table_file):
        guid2instance = {}
        guid2programs = {}
        guid_lst = []
        id2table = {}
        instances = json.load(open(predict_file, "r"))
        tables = json.load(open(table_file, "r"))
        for instance in instances:
            guid = int(instance['guid'])
            if self.mode == self.TRAIN_MODE and instance.get('programs', []):
                guid2programs[guid] = instance.pop('programs')
            guid2instance[guid] = instance
            guid_lst.append(guid)
        for table in tables:
            id2table[table['id']] = table
        return guid_lst, guid2instance, guid2programs, id2table

    def execute_augmented_programs(self):
        for guid in self.guid_lst:
            for program in self.guid2programs.get(guid, []):
                program['subsampled'] = False if self.config.subsample else True
        if self.local_rank in [-1, 0]:
            single_candidate_cnt = 0
            for _, programs in self.guid2programs.items():
                if len(programs) == 1:
                    single_candidate_cnt += 1
            print("Data statistics")
            print("\ttotal_instance({}) average_candidate_cnt({:.3f}) single_candidate_({})".format(len(self.guid2programs), np.mean([len(programs) for guid, programs in self.guid2programs.items()]), single_candidate_cnt), flush=True)
            for key in ['logic_form_acc', 'exe_acc']:
                lst = []
                for guid, programs in self.guid2programs.items():
                    lst.append(max(program[key] for program in programs))
                print("\t{}: avg({:.4f}) max({:.4f}) min({:.4f})".format(key, np.mean(lst), np.max(lst), np.min(lst)), flush=True)

    def _read_guid2programs(self, guid2programs_file):
        guid2programs = {}
        dirname, prefix = os.path.split(guid2programs_file.rstrip('/'))
        for fname in os.listdir(dirname):
            if fname.startswith(prefix):
                data = json.load(open(os.path.join(dirname, fname), "r"))
                for guid, programs in data.items():
                    guid = int(guid)
                    guid2programs[guid] = programs
        return guid2programs

    def get_guid2programs(self):
        return self.guid2programs

    def clean_guid2programs(self):
        self.guid2programs = {}

    def get_best_program_results(self, criterion='answer'):
        '''
        *   Arguments:
            *   criterion (str): 'answer' or 'gen_prob'
        '''
        res = {'pred_program': {}}
        for guid in self.guid_lst:
            programs = self.guid2programs.get(guid, [])
            if programs:
                best_score = [0, 0] if criterion == 'answer' else -np.inf
                best_program = None
                for item in programs:
                    prob_key = 'prior_log_prob' if 'prior_log_prob' in item else 'post_log_prob'
                    score = [item[prob_key], item['exe_acc']] if criterion == 'answer' else item[prob_key]
                    if score > best_score:
                        best_score = score
                        best_program = item
                if best_program is not None:
                    res['pred_program'][guid] = best_program
        return res

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
            for guids in self._shard_dataloader:
                for guid in guids[0].cpu().numpy().tolist():
                    if guid in self.guid2programs:
                        local_shard_guid2programs[guid] = self.guid2programs[guid]
            json.dump(local_shard_guid2programs, open(local_filename, "w"))
            filenames = [os.path.join(self.config.cache_dir, filename + str(local_rank)) for local_rank in range(dist.get_world_size())]
            while not all(os.path.exists(fname) and time.time() - os.path.getmtime(fname) > 2 for fname in filenames):
                pass
            filename = os.path.join(self.config.cache_dir, filename)
            if self.local_rank == 0:
                print("Synchronizing data:", self.local_rank, flush=True)
                agg_local_shard_guid2programs = {}
                for fname in filenames:
                    data = {int(guid): programs for guid, programs in json.load(open(fname, "r", encoding='utf-8')).items()}
                    agg_local_shard_guid2programs.update(data)
                self.guid2programs.update(agg_local_shard_guid2programs)
                for fname in filenames:
                    os.remove(fname)
                json.dump(agg_local_shard_guid2programs, open(filename, "w", encoding='utf-8'))
                num_replicas = dist.get_world_size()
                while not all(os.path.exists(os.path.join(self.config.cache_dir, 'rank_{}_sync_{}_{}_ok'.format(local_rank + 1, self.epoch, self._shard_id))) for local_rank in range(num_replicas - 1)):
                    pass
                for local_rank in range(num_replicas - 1):
                    os.remove(os.path.join(self.config.cache_dir, "rank_{}_sync_{}_{}_ok".format(local_rank + 1, self.epoch, self._shard_id)))
            else:
                print("Waiting:", self.local_rank, flush=True)
                while not (os.path.exists(filename) and time.time() - os.path.getmtime(filename) > 2):
                    pass
                print("Loading:", self.local_rank, flush=True)
                self.guid2programs.update({int(guid): programs for guid, programs in json.load(open(filename, "r", encoding='utf-8')).items()})
                os.system("touch {}/rank_{}_sync_{}_{}_ok".format(self.config.cache_dir, self.local_rank, self.epoch, self._shard_id))
            print("Exit sync:", self.local_rank, flush=True)

    def set_current_batch_guids(self, guids):
        if isinstance(guids, (list, tuple)):
            self.guids = guids
        else:
            self.guids = guids.cpu().numpy().tolist()

    def infer_program(self, instances, posterior_net=None, infer=True, execute=True, compute_log_prob=True, execution_guided=False, augmented=False):
        '''
        *   Arguments:
            *   infer (bool): whether to infer programs from posterior_net
                *   if infer == False, each instance should have ^sql^ and ^query^
            *   execute (boo): whether to compute execution results
            *   compute_log_prob (bool): whether to compute log probs of programs
        '''
        assert not execution_guided

        if infer or compute_log_prob:
            assert posterior_net is not None
            posterior_net.eval()

        results = []
        for start in range(0, len(instances), self.config.batch_size):
            batch = instances[start: start + self.config.batch_size]

            if not infer:
                for instance in batch:
                    # assert 'sql' in instance and 'query' in instance
                    assert 'sql' in instance
                    assert 'wvi_corenlp' in instance
                inps = self.convert_features_to_inputs_for_Posterior(batch, is_train=True)
                pr_wvi_corenlp = inps['list_features']['wvi_corenlp']
                pr_sql_i = inps['list_features']['sql']
                pr_sc = inps['list_features']['select_col']
                pr_sa = inps['list_features']['agg_op']
                pr_wn = inps['list_features']['num_conds']
                pr_wc = inps['list_features']['where_cols']
                pr_wo = inps['list_features']['where_ops']
                pr_wvi = inps['list_features']['where_values']

            for instance in batch:
                assert 'ground_truth_program' in instance
                instance['sql'] = instance['ground_truth_program']['sql']
                instance['wvi_corenlp'] = instance['ground_truth_program']['wvi_corenlp']

            inps = self.convert_features_to_inputs_for_Posterior(batch, is_train=execute)
            nlu = inps['list_features']['question']
            nlu_t = inps['list_features']['question_tok']
            nlu_tt = inps['list_features']['question_wps']
            tt_to_t_idx = inps['list_features']['wp2token_index']
            tb = inps['list_features']['table']
            if execute:
                sql_i = inps['list_features']['sql']
                g_sc = inps['list_features']['select_col']
                g_sa = inps['list_features']['agg_op']
                g_wn = inps['list_features']['num_conds']
                g_wc = inps['list_features']['where_cols']
                g_wo = inps['list_features']['where_ops']
                g_wvi = inps['list_features']['where_values']

            if infer:
                # No Execution guided decoding
                res = posterior_net(**inps['tensor_inps'])
                s_sc = res['batch_id2logit_select_col']
                s_sa = res['batch_id2logit_agg_op']
                s_wn = res['batch_id2logit_num_conds']
                s_wc = res['batch_id2logits_where_cols']
                s_wo = res['batch_id2logits_where_ops']
                s_wv = res['batch_id2logits_where_values']

                # prediction
                pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = utils_wikisql.pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv)
                pr_wv_str, pr_wv_str_wp, pr_wvi_corenlp = utils_wikisql.convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu, return_wvi_tok=True)
                pr_sql_i = utils_wikisql.generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu)

            if compute_log_prob:
                for instance, sql, wvi_corenlp in zip(batch, pr_sql_i, pr_wvi_corenlp):
                    instance['sql'] = sql
                    instance['wvi_corenlp'] = wvi_corenlp
                    # instance['query'] = sql # unused
                _inps = self.convert_features_to_inputs_for_Posterior(batch, is_train=True)
                _inps['tensor_inps'].update(_inps['program_inps'])
                log_prob = posterior_net(**_inps['tensor_inps'])['batch_id2program_log_probs'].cpu().numpy().tolist()

            # g_sql_q = generate_sql_q(sql_i, tb)
            # pr_sql_q = generate_sql_q(pr_sql_i, tb)

            if execute:
                cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
                cnt_wc1_list, cnt_wo1_list, \
                cnt_wvi1_list, cnt_wv1_list = utils_wikisql.get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi, pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, sql_i, pr_sql_i, mode='test')

                cnt_lx1_list = utils_wikisql.get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list)

                # Execution accura y test
                cnt_x1_list = []
                # lx stands for logical form accuracy

                # Execution accuracy test.
                if not augmented:
                    cnt_x1_list = []
                    for b, instance in enumerate(batch):
                        g_ans = instance.get('answer', '')
                        pr_ans = self.engine.execute(instance['table_id'], pr_sc[b], pr_sa[b], pr_sql_i[b]['conds'])
                        if pr_ans == g_ans:
                            cnt_x1_list.append(1)
                        else:
                            cnt_x1_list.append(0)
                    # cnt_x1_list, g_ans, pr_ans = utils_wikisql.get_cnt_x_list(self.engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)
                else:
                    cnt_x1_list = [1] * len(batch)

            for b, (pr_sql_i1, wvi_corenlp) in enumerate(zip(pr_sql_i, pr_wvi_corenlp)):
                results1 = {
                    'sql': {
                        'sel': int(pr_sql_i1['sel']),
                        'agg': int(pr_sql_i1['agg']),
                        'conds': [[int(cond[0]), int(cond[1]), cond[2] if isinstance(cond[2], str) else (int(cond[2]) if isinstance(cond[2], int) else float(cond[2]))] for cond in pr_sql_i1.get('conds', [])]
                    },
                    'wvi_corenlp': [[int(inv[0]), int(inv[1])] for inv in wvi_corenlp]
                }
                if compute_log_prob:
                    results1['post_log_prob'] = log_prob[b]
                else:
                    results1['post_log_prob'] = -np.inf
                if execute:
                    for key, val in zip(
                        ['select_col_acc', 'agg_op_acc', 'num_conds_acc', 'where_cols_acc', 'where_ops_acc', 'where_values_acc', 'logic_form_acc', 'exe_acc'],
                        [cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_lx1_list, cnt_x1_list]
                    ):
                        results1[key] = int(val[b])
                results.append(results1)
        return results

    def execute_program(self, instance, augmented=False):
        return self.infer_program([instance], infer=False, execute=True, compute_log_prob=False, augmented=augmented)[0]

    def subsample_programs(self, guids):
        if not self.config.subsample:
            return
        cnts = np.array([len([program for program in self.guid2programs.get(guid, []) if program['augmented']]) for guid in guids])
        tot = 30 * len(guids)
        n_each = 50
        while sum(cnts) > tot:
            cnts = np.minimum(cnts, n_each)
            n_each -= 10
        for guid, cnt in zip(guids, cnts):
            if cnt == 0:
                continue
            indices = []
            for pid, program in enumerate(self.guid2programs[guid]):
                if program['augmented']:
                    program['subsampled'] = False
                    indices.append(pid)
            for pid in np.random.choice(indices, cnt, replace=False):
                self.guid2programs[guid][pid]['subsampled'] = True

    def update_programs(self, posterior_net, prior_net, guids=None, program_samples=None, compute_log_prob=True, subsample=True):
        '''
        *   Arguments:
            *   program_samples (torch.LongTensor): [batch, num_evidence_samples * num_sketch_samples * num_filling_samples, max_program_len]
        '''
        if guids is None:
            guids = self.guids

        update_new_candidates = (program_samples is not None)
        if update_new_candidates:
            assert len(guids) == len(program_samples)
        
        instances = []
        batch_id2num_program = []
        if self.mode == self.TRAIN_MODE and subsample:
            self.subsample_programs(guids)
        for batch_id, guid in enumerate(guids):
            if update_new_candidates:
                sample = program_samples[batch_id]
                if guid not in self.guid2programs or all(sample['sql'] != program['sql'] for program in self.guid2programs[guid]):
                    if guid not in self.guid2programs:
                        self.guid2programs[guid] = []
                    sample['augmented'] = False
                    sample['correct'] = bool(sample['exe_acc'] >= self.config.exe_acc_threshold_for_sample)
                    self.guid2programs[guid].append(sample)

            num_programs = len([program for program in self.guid2programs.get(guid, []) if not program['augmented'] or program['subsampled']])
            batch_id2num_program.append(num_programs)
            if num_programs > 0:
                for program in self.guid2programs[guid]:
                    if not program['augmented'] or program['subsampled']:
                        instance = copy.deepcopy(self.guid2instance[guid])
                        instance['sql'] = program['sql']
                        instance['wvi_corenlp'] = program['wvi_corenlp']
                        instances.append(instance)

        # compute log_prob of generating the question given the passage & program
        if sum(batch_id2num_program) > 0 and compute_log_prob:
            for model in [posterior_net, prior_net]:
                if model is None:
                    continue
                model.eval()
                if hasattr(model, 'module'):
                    is_prior = model.module.is_prior
                else:
                    is_prior = model.is_prior

                with torch.no_grad():
                    scores = {
                        'post_log_prob': [],
                        'prior_log_prob': []
                    }
                    for idx in range(0, sum(batch_id2num_program), self.config.batch_size):
                        batch = instances[idx: idx + self.config.batch_size]
                        if is_prior:
                            inps = self.convert_features_to_inputs_for_Prior(batch)
                        else:
                            inps = self.convert_features_to_inputs_for_Posterior(batch, is_train=True)
                            inps['tensor_inps'].update(inps['program_inps'])
                            inps = inps['tensor_inps']
                        res = model(**inps)
                        if is_prior:
                            scores['prior_log_prob'].extend(res['log_prob'].cpu().numpy().tolist())
                        else:
                            scores['post_log_prob'].extend(res['batch_id2program_log_probs'].cpu().numpy().tolist())
                score_iter = {key: iter(val) for key, val in scores.items()}
                for batch_id, (start_idx, cnt) in enumerate(list(zip(np.cumsum([0] + batch_id2num_program), batch_id2num_program))):
                    guid = guids[batch_id]
                    if cnt == 0:
                        continue
                    for item in self.guid2programs[guid]:
                        if not item['augmented'] or item['subsampled']:
                            if is_prior:
                                for key, iterator in score_iter.items():
                                    if key.startswith('prior'):
                                        item[key] = next(iterator)
                            else:
                                item['post_log_prob'] = next(score_iter['post_log_prob'])
        metrics = {}
        for guid in guids:
            programs = [program for program in self.guid2programs[guid] if not program['augmented'] or program['subsampled']]
            if programs:
                for key in [key for key, val in programs[0].items() if isinstance(val, (int, float, np.int, np.int32, np.int64, np.float, np.float32, np.float64))]:
                    metrics[key] = metrics.get(key, 0) + max(program.get(key, 0) for program in programs)
        for prefix, cmp_key, model in zip(
            ['prior', 'posterior'], ['prior_log_prob', 'post_log_prob'], [prior_net, posterior_net],
        ):
            if model is not None:
                metrics = self._stat_hit(metrics, cmp_key, prefix)
        if metrics:
            for key, val in metrics.items():
                metrics[key] = val / len(guids)
        return metrics

    def _stat_hit(self, metrics, cmp_key, prefix):
        for guid in self.guids:
            programs = [program for program in self.guid2programs[guid] if not program['augmented'] or program['subsampled']]
            assert programs
            programs = sorted(programs, key=lambda x: x[cmp_key])
            program_set = [self.sql_to_str(prog['sql']) for prog in programs]
            g = self.guid2instance[guid]['ground_truth_program']['sql']
            g_conds = sorted([[cond[0], cond[1], str(cond[2]).lower()] for cond in g.get('conds', [])])
            g_wc = [cond[0] for cond in g_conds]
            g_wo = [cond[1] for cond in g_conds]
            g_wv = [cond[2] for cond in g_conds]
            p = programs[-1]['sql']
            p_conds = sorted([[cond[0], cond[1], str(cond[2]).lower()] for cond in p.get('conds', [])])
            p_wc = [cond[0] for cond in p_conds]
            p_wo = [cond[1] for cond in p_conds]
            p_wv = [cond[2] for cond in p_conds]
            gt = self.sql_to_str(g)

            hit = int(program_set[-1] == gt)
            hit_at3 = int(gt in program_set[-3:])
            hit_at5 = int(gt in program_set[-5:])
            has_gt = int(gt in program_set)
            metrics['{}_hit_at1'.format(prefix)] = metrics.get('{}_hit_at1'.format(prefix), 0) + hit
            metrics['{}_hit_at3'.format(prefix)] = metrics.get('{}_hit_at3'.format(prefix), 0) + hit_at3
            metrics['{}_hit_at5'.format(prefix)] = metrics.get('{}_hit_at5'.format(prefix), 0) + hit_at5
            metrics['{}_hit_sel'.format(prefix)] = metrics.get('{}_hit_sel'.format(prefix), 0) + int(g['sel'] == p['sel'])
            metrics['{}_hit_agg'.format(prefix)] = metrics.get('{}_hit_agg'.format(prefix), 0) + int(g['agg'] == p['agg'])
            metrics['{}_hit_where_col'.format(prefix)] = metrics.get('{}_hit_where_col'.format(prefix), 0) + int(g_wc == p_wc)
            metrics['{}_hit_where_op'.format(prefix)] = metrics.get('{}_hit_where_op'.format(prefix), 0) + int(g_wo == p_wo)
            metrics['{}_hit_where_value'.format(prefix)] = metrics.get('{}_hit_where_value'.format(prefix), 0) + int(g_wv == p_wv)
            metrics['{}_hit_gt'.format(prefix)] = metrics.get('{}_hit_gt'.format(prefix), 0) + has_gt
        return metrics

    def sql_to_str(self, sql):
        return "select {} agg {} conds {}".format(sql['sel'], sql['agg'], " ".join(sorted([str([cond[0], cond[1], str(cond[2]).lower()]) for cond in sql.get('conds', [])])))

    def _random_sample(self, programs, prob_key='post_log_prob'):
        log_probs = np.array([max(item[prob_key], self.config.clamp_log_prob_min) for item in programs])
        probs = np.exp(log_probs)
        probs = probs / np.sum(probs)
        idx = np.random.choice(list(range(len(probs))), size=1, p=probs)[0]
        return programs[idx]

    def _greedy_sample(self, programs, prob_key='prior_log_prob'):
        best_program = None
        for program in programs:
            if best_program is None or program[prob_key] > best_program[prob_key]:
                best_program = program
        return best_program

    def _sample_program(self, sample_type_for_prior, sample_type_for_posterior, use_other_net_for_selection, sample_from_augmented_programs_only=True):
        '''
        *   Arguments:
            *   sample_type_for_prior (str): 'random' or 'greedy'
            *   sample_type_for_posterior (str): 'random' or 'greedy'
        '''
        for batch_id in range(len(self.guids)):
            guid = self.guids[batch_id]
            if self.guid2programs.get(guid, []):
                programs = [program for program in self.guid2programs[guid] if (program['augmented'] and program['subsampled']) or (not program['augmented'] and program['correct'] and not sample_from_augmented_programs_only)]
                if sample_type_for_prior is not None:
                    prob_key = 'post_log_prob' if use_other_net_for_selection else 'prior_log_prob'
                    sample_for_prior = self._random_sample(programs, prob_key=prob_key) if sample_type_for_prior == 'random' else self._greedy_sample(programs, prob_key=prob_key)
                    self.guid2prior_sample[guid] = sample_for_prior
                if sample_type_for_posterior is not None:
                    prob_key = 'prior_log_prob' if use_other_net_for_selection else 'post_log_prob'
                    sample_for_posterior = self._random_sample(programs, prob_key=prob_key) if sample_type_for_posterior == 'random' else self._greedy_sample(programs, prob_key=prob_key)
                    self.guid2posterior_sample[guid] = sample_for_posterior

    def sample_program(self, for_posterior, use_other_net_for_selection):
        raise NotImplementedError()

    def get_posterior_network_inputs(self, with_sample):
        instances = []
        for guid in self.guids:
            instance = self.guid2instance[guid]
            if with_sample:
                sample = self.guid2posterior_sample[guid]
            else:
                sample = instance['ground_truth_program']
            instance['sql'] = sample['sql']
            instance['wvi_corenlp'] = sample['wvi_corenlp']
            instances.append(instance)
        return self.convert_features_to_inputs_for_Posterior(instances, is_train=True)

    def get_prior_network_inputs(self, with_sample):
        assert with_sample
        instances = []
        for guid in self.guids:
            sample = self.guid2prior_sample[guid]
            instance = copy.deepcopy(self.guid2instance[guid])
            for key in ['sql', 'wvi_corenlp']:
                instance[key] = sample[key]
            instances.append(instance)
        inps = self.convert_features_to_inputs_for_Prior(instances)
        return inps

    def _convert_post_features_to_prior_format(self, question_tokens, passage_tokens, col2passage_index_range):
        post_tokenizer = self.library.post_tokenizer
        prior_tokenizer = self.library.prior_tokenizer
        _question_tokens = prior_tokenizer.tokenize(post_tokenizer.convert_tokens_to_string(question_tokens))
        _col2passage_index_range = {}
        _passage_tokens = []
        cols = []
        for i in range(len(col2passage_index_range)):
            inv = col2passage_index_range[str(i)]
            col = post_tokenizer.convert_tokens_to_string(passage_tokens[inv[0]: inv[1]])
            col = prior_tokenizer.tokenize(" " + col)
            _col2passage_index_range[str(i)] = [len(_passage_tokens), len(_passage_tokens) + len(col)]
            _passage_tokens.extend(col + ['[col]'])
        if _passage_tokens:
            _passage_tokens = _passage_tokens[:-1]
        return _question_tokens, _passage_tokens, _col2passage_index_range

    def convert_features_to_inputs_for_Prior(self, instances):
        input_ids = []
        attention_mask = []
        decoder_input_ids = []
        post_tokenizer = self.library.post_tokenizer
        prior_tokenizer = self.library.prior_tokenizer

        for instance in instances:
            # question_tokens = instance['question_tokens']
            # passage_tokens = instance['passage_tokens']
            # col2passage_index_range = instance['col2passage_index_range']
            question_tokens, passage_tokens, col2passage_index_range = self._convert_post_features_to_prior_format(instance['question_tokens'], instance['passage_tokens'], instance['col2passage_index_range'])
            sql = instance['sql']
            qlen = len(question_tokens)
            
            sql_tokens, spans = self.library.decode_program(sql)

            input_tokens = ['<s>', '[tab]'] + passage_tokens + ['[prog]'] + sql_tokens + ['</s>']
            input_ids.append(self.library.convert_tokens_to_ids(input_tokens, is_prior=True))
            decoder_input_ids.append(self.library.convert_tokens_to_ids(['<s>'] + question_tokens + ['</s>'], is_prior=True))
            attn_map = [[1.0] * len(input_ids[-1])] + \
                [[1.0] * (len(passage_tokens) + 2) + [0.0] * (len(sql_tokens) + 1) + [1.0]] * (len(passage_tokens) + 1) + \
                [[1.0] + [0.0] * (len(passage_tokens) + 1) + [1.0] * (len(sql_tokens) + 2) for _ in range(len(sql_tokens) + 1)] + \
                [[1.0] * len(input_ids[-1])]
            for span in spans:
                assert span[0] == 'table'
                col = span[1]
                inv = col2passage_index_range[str(col)]
                inv = [inv[0] + 2, inv[1] + 2]
                idx = len(passage_tokens) + 3 + span[2]
                tmp = [0.0] * len(input_ids[-1])
                for i in range(inv[0], inv[1]):
                    tmp[i] = 1.0
                attn_map[idx] = tmp
            attention_mask.append(attn_map)

        _inps = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids
        }

        inps = {}
        for key, value in _inps.items():
            if key.endswith('_ids'):
                inps[key] = pad_sequence([torch.tensor(line, dtype=torch.long) for line in value], batch_first=True, padding_value=(self.library.get_pad_token_id(is_prior=True) if key.endswith('input_ids') else 0))
            elif key == 'attention_mask':
                tmp = torch.empty((len(value), max(len(plane) for plane in value), max(len(line) for plane in value for line in plane)), dtype=torch.float32).fill_(0.0)
                for row, plane in enumerate(value):
                    plane = torch.tensor(plane, dtype=torch.float32)
                    tmp[row, :plane.size(0), :plane.size(1)] = plane
                inps[key] = tmp
        return inps

    def convert_features_to_inputs_for_Posterior(self, instances, is_train=True):
        for instance in instances:
            instance['query'] = ""
        if is_train:
            question, question_tokens, sql_i, sql_q, sql_t, table, header_tokens, headers, wvi_corenlp = utils_wikisql.get_fields(instances, self.id2table, train=is_train, no_hs_t=True, no_sql_t=True)
        else:
            question, question_tokens, sql_i, sql_q, sql_t, table, header_tokens, headers, wvi_corenlp = utils_wikisql.get_fields(instances, self.id2table, train=is_train, no_hs_t=True, no_sql_t=True)

        input_ids, input_mask, segment_ids,\
            _, question_index_range, headers_index_range,\
                num_question_wps, header_wps_len, num_headers,\
                    question_wps, token2wp_index, wp2token_index\
                        = utils_wikisql.get_inputs(self.library.post_tokenizer, question_tokens, headers, 512)

        question_indices = pad_sequence([torch.tensor(list(range(inv[0], inv[1])), dtype=torch.long) for inv in question_index_range], batch_first=True, padding_value=-1)
        header_indices = []
        for batch_id, header_index_range in enumerate(headers_index_range):
            for inv in header_index_range:
                header_indices.append(list(range(inv[0] + batch_id * input_ids.size(1), inv[1] + batch_id * input_ids.size(1))))
        header_indices = pad_sequence([torch.tensor(indices, dtype=torch.long) for indices in header_indices], batch_first=True, padding_value=-1)

        if is_train:
            select_col_lst, agg_op_lst, num_conds_lst, where_cols_lst, where_ops_lst, _ = utils_wikisql.get_g(sql_i)

            where_values_lst = utils_wikisql.get_g_wvi_bert_from_g_wvi_corenlp(token2wp_index, wvi_corenlp)

            select_col_lst = [item[0] for item in select_col_lst]
            agg_op_lst = [item[0] for item in agg_op_lst]
            num_conds_lst = [item[0] for item in num_conds_lst]
            where_cols_lst = [item[0] for item in where_cols_lst]
            where_ops_lst = [item[0] for item in where_ops_lst]
            where_values_lst = [item[0] for item in where_values_lst]

            max_num_conds = max(num_conds_lst)
            where_cols = torch.zeros((len(where_cols_lst), max_num_conds), dtype=torch.long)
            where_ops = torch.zeros((len(where_ops_lst), max_num_conds), dtype=torch.long)
            where_values = torch.zeros((len(where_values_lst), max_num_conds, 2), dtype=torch.long)
            for batch_id, n in enumerate(num_conds_lst):
                if n == 0:
                    continue
                where_cols[batch_id, :n] = torch.tensor(where_cols_lst[batch_id], dtype=torch.long)
                where_ops[batch_id, :n] = torch.tensor(where_ops_lst[batch_id], dtype=torch.long)
                where_values[batch_id, :n] = torch.tensor(where_values_lst[batch_id], dtype=torch.long)

        inps = {
            'tensor_inps': {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,

                'question_indices': question_indices,
                'header_indices': header_indices,

                'num_question_tokens': torch.tensor(num_question_wps, dtype=torch.long),
                'num_header_tokens': torch.tensor(header_wps_len, dtype=torch.long),
                'num_headers': torch.tensor(num_headers, dtype=torch.long)
            },
            'list_features': {
                'question': question,
                'question_tok': question_tokens,
                'question_wps': question_wps,
                'wp2token_index': wp2token_index,
                'table': table,

                'sql': [item[0] for item in sql_i] if is_train else None,
                'wvi_corenlp': [item[0] for item in wvi_corenlp] if is_train else None,
                'select_col': select_col_lst if is_train else None,
                'agg_op': agg_op_lst if is_train else None,
                'num_conds': num_conds_lst if is_train else None,
                'where_cols': where_cols_lst if is_train else None,
                'where_ops': where_ops_lst if is_train else None,
                'where_values': where_values_lst if is_train else None
            },
            'program_inps': {
                'select_col': torch.tensor(select_col_lst, dtype=torch.long),
                'agg_op': torch.tensor(agg_op_lst, dtype=torch.long),
                'num_conds': torch.tensor(num_conds_lst, dtype=torch.long),
                'where_cols': where_cols,
                'where_ops': where_ops,
                'where_values': where_values
            } if is_train else {}
        }
        return inps
