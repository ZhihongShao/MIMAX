import numpy as np
from CustomDataset import CustomDatasetConfig, CustomDataset

class MemoisedDatasetConfig(CustomDatasetConfig):
    def __init__(self):
        super().__init__()
        self.alpha = 0.5
        self.gamma = 0.5
        self.loss_type = ""

    def set_loss_type(self, loss_type):
        self.loss_type = loss_type

class MemoisedDataset(CustomDataset):
    def __init__(self, predict_file, library, mode, local_rank, num_shards=1, config=MemoisedDatasetConfig()):
        super().__init__(predict_file, library, mode, local_rank, num_shards=num_shards, config=config)

    def increment_epoch(self):
        super().increment_epoch()
        if self.config.loss_type == 'hard-em-thres':
            self.thres_epoch += 1

    def _get_trainable_indices(self, indices):
        if self.config.loss_type == 'hard-em-thres':
            if not hasattr(self, 'thres_epoch'):
                self.thres_epoch = self.epoch
            trainable_indices = []
            prog_log_prob_thres = np.log(self.config.alpha * self.config.gamma ** self.thres_epoch)
            positive_indices = []
            negative_indices = []
            for idx in indices:
                if idx % 2 == 0:
                    guid = self.positive_guids[int(idx / 2)]
                    if not self.guid2programs.get(guid, []):
                        continue
                    programs = [program for program in self.guid2programs[guid]]
                    if max(program['post_log_prob'] for program in programs) > prog_log_prob_thres:
                        positive_indices.append(idx)
                else:
                    negative_indices.append(idx)
            if len(positive_indices) < int(0.25 * len(indices)):
                if self.local_rank in [-1, 0]:
                    print("NUM POS EXAM {}".format(len(positive_indices)), flush=True)
                    print("THRESHOLD {:.6f}".format(prog_log_prob_thres), flush=True)
                    print("NO TRAINABLE GUIDS FOUND FOR EPOCH {}".format(self.thres_epoch), flush=True)
                    print("INCREMENT EPOCH AND START ANOTHER TRIAL", flush=True)
                self.thres_epoch += 1
                return self._get_trainable_indices(indices)
            trainable_indices = positive_indices + np.random.choice(negative_indices, len(positive_indices), replace=False).tolist()
        else:
            trainable_indices = indices
        return trainable_indices

    def sample_program(self, for_posterior, use_other_net_for_selection):
        if for_posterior:
            self._sample_program(sample_type_for_prior=None, sample_type_for_posterior='greedy', use_other_net_for_selection=use_other_net_for_selection)
        else:
            self._sample_program(sample_type_for_prior='random', sample_type_for_posterior=None, use_other_net_for_selection=use_other_net_for_selection)
