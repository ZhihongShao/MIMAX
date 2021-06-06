import json

import numpy as np

from CustomDataset import CustomDatasetConfig, CustomDataset

class NeRdDatasetConfig(CustomDatasetConfig):
    def __init__(self):
        super().__init__()
        self.exe_acc_threshold_for_sample = 0.5
        self.alpha = 0.5
        self.gamma = 0.5
        self.loss_type = ""
        self.extreme_case = True

    def set_loss_type(self, loss_type):
        self.loss_type = loss_type

class NeRdDataset(CustomDataset):
    def __init__(self, predict_file, table_file, db_file, library, mode, local_rank, num_shards=1, guid2programs_file=None, config=NeRdDatasetConfig()):
        super().__init__(predict_file, table_file, db_file, library, mode, local_rank, num_shards=num_shards, guid2programs_file=guid2programs_file, config=config)

    def _read_examples(self, predict_file, table_file):
        if self.mode == self.TRAIN_MODE and self.config.loss_type == 'mml':
            guid = 0
            guid2instance = {}
            guid2programs = {}
            guid_lst = []
            id2table = {}
            instances = json.load(open(predict_file, "r"))
            tables = json.load(open(table_file, "r"))
            for instance in instances:
                if instance.get('programs', []):
                    programs = instance.pop('programs')
                    for program in programs:
                        guid2instance[guid] = instance
                        guid2programs[guid] = [program]
                        guid_lst.append(guid)
                        guid += 1
            for table in tables:
                id2table[table['id']] = table
            return guid_lst, guid2instance, guid2programs, id2table
        else:
            guid_lst, guid2instance, guid2programs, id2table = super()._read_examples(predict_file, table_file)
            if self.mode == self.TRAIN_MODE:
                guid_lst = [guid for guid in guid_lst if guid in guid2programs]
            if self.mode == self.TRAIN_MODE and self.config.extreme_case:
                guid2cand_cnt = []
                for guid, programs in guid2programs.items():
                    guid2cand_cnt.append((guid, len(programs)))
                guid2cand_cnt = sorted(guid2cand_cnt, key=lambda x: x[1])
                _guid_lst = []
                _guid2instance = {}
                _guid2programs = {}
                for item in guid2cand_cnt[-10000:]:
                    guid = item[0]
                    _guid_lst.append(guid)
                    _guid2instance[guid] = guid2instance[guid]
                    _guid2programs[guid] = guid2programs[guid]
                return _guid_lst, _guid2instance, _guid2programs, id2table
            return guid_lst, guid2instance, guid2programs, id2table

    def _get_trainable_guids(self, guids):
        if self.config.loss_type in ['mml', 'hard-em']:
            trainable_guids = [guid for guid in guids if self.guid2programs.get(guid, [])]
            return trainable_guids
        elif self.config.loss_type == 'hard-em-thres':
            trainable_guids = []
            prog_log_prob_thres = np.log(self.config.alpha * self.config.gamma ** self.epoch)
            for guid in guids:
                if not self.guid2programs.get(guid, []):
                    continue
                programs = [program for program in self.guid2programs[guid] if program['subsampled']]
                if (len(programs) == 1 and self.epoch == 0) or max(program['post_log_prob'] for program in programs) > prog_log_prob_thres:
                    trainable_guids.append(guid)
            if len(trainable_guids) < int(0.5 * len(guids)):
                if self.local_rank in [-1, 0]:
                    print("NO TRAINABLE GUIDS FOUND FOR EPOCH {}".format(self.epoch), flush=True)
                    print("INCREMENT EPOCH AND START ANOTHER TRIAL", flush=True)
                self.epoch += 1
                return self._get_trainable_guids(guids)
            else:
                return trainable_guids
        else:
            raise RuntimeError()

    def sample_program(self, for_posterior, use_other_net_for_selection):
        self._sample_program(sample_type_for_prior=None, sample_type_for_posterior='greedy', use_other_net_for_selection=use_other_net_for_selection)
