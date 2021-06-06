import os

import torch
import torch.nn as nn

class GenerativeBaseModel(nn.Module):
    def __init__(self, library, is_prior):
        super().__init__()
        self.library = library
        self.is_prior = is_prior

    def _create_position_ids_from_input_ids(self, input_ids):
        mask = input_ids.ne(self.library.get_pad_token_id(is_prior=self.is_prior)).int()
        incremental_indicies = (torch.cumsum(mask, dim=-1).type_as(mask) - 1) * mask
        return incremental_indicies.long()

    def save(self, ckpt_dir, save_to_best_dir, global_step):
        '''
        *   Returns:
            *   the name of the directory where the model is saved
        '''
        path = os.path.join(ckpt_dir, "Prior" if self.is_prior else "Posterior", "best" if save_to_best_dir else "tmp", str(global_step))
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, "pytorch_model_{}.bin".format(global_step))
        torch.save(self.state_dict(), filename)
        return path

    def restore(self, ckpt_dir, restore_best, map_location=None):
        '''
        *   Arguments:
            *   restore_best (boolean):
                *   if ^True^, restore the best checkpoint
                *   if ^False^, restore the latest checkpoint
        *   Returns:
            *   global_step (int)
            *   the name of the directory where the restored model is save
        '''
        path = os.path.join(ckpt_dir, "Prior" if self.is_prior else "Posterior")
        best_dir = os.path.join(path, "best")
        tmp_dir = os.path.join(path, "tmp")
        if not os.listdir(best_dir):
            return -1, None
        best_global_step = max(int(name) for name in os.listdir(best_dir))
        if restore_best:
            loaded = torch.load(os.path.join(best_dir, str(best_global_step), "pytorch_model_{}.bin".format(best_global_step)), map_location=map_location)
            self.load_state_dict(loaded)
            return best_global_step, os.path.join(best_dir, str(best_global_step))
        else:
            if os.listdir(tmp_dir):
                tmp_global_step = max(int(name) for name in os.listdir(tmp_dir))
            else:
                tmp_global_step = -1
            if best_global_step > tmp_global_step:
                latest_global_step = best_global_step
                dirname = best_dir
            else:
                latest_global_step = tmp_global_step
                dirname = tmp_dir
            loaded = torch.load(os.path.join(dirname, str(latest_global_step), "pytorch_model_{}.bin".format(latest_global_step)), map_location=map_location)
            self.load_state_dict(loaded)
            return latest_global_step, os.path.join(dirname, str(latest_global_step))
