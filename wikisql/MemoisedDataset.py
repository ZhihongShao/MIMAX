import os
import json
import re
import time

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from CustomDataset import CustomDatasetConfig, CustomDataset

class MemoisedDatasetConfig(CustomDatasetConfig):
    def __init__(self):
        super().__init__()
        self.exe_acc_threshold_for_sample = 0.5
        self.loss_type = ""
        self.extreme_case = True

    def set_loss_type(self, loss_type):
        self.loss_type = loss_type

class MemoisedDataset(CustomDataset):
    def __init__(self, predict_file, table_file, db_file, library, mode, local_rank, num_shards=1, guid2programs_file=None, config=MemoisedDatasetConfig()):
        super().__init__(predict_file, table_file, db_file, library, mode, local_rank, num_shards=num_shards, guid2programs_file=guid2programs_file, config=config)

    def _read_examples(self, predict_file, table_file):
        guid_lst, guid2instance, guid2programs, id2table = super()._read_examples(predict_file, table_file)
        if self.mode == self.TRAIN_MODE and self.config.loss_type == 'mimax':
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
        trainable_guids = []
        for guid in guids:
            if self.config.loss_type == 'mimax' and len([program for program in self.guid2programs.get(guid, []) if program['augmented']]):
                trainable_guids.append(guid)
        return trainable_guids

    def sample_program(self, for_posterior, use_other_net_for_selection):
        if for_posterior:
            self._sample_program(sample_type_for_prior=None, sample_type_for_posterior='greedy', use_other_net_for_selection=use_other_net_for_selection, sample_from_augmented_programs_only=(self.config.loss_type == 'mimax'))
        else:
            self._sample_program(sample_type_for_prior='random', sample_type_for_posterior=None, use_other_net_for_selection=use_other_net_for_selection, sample_from_augmented_programs_only=(self.config.loss_type == 'mimax'))
