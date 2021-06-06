import os

import torch
import torch.nn as nn

from transformers import BertConfig

from . import GenerativeBaseModel
from .modeling_bert import BertForQuestionAnswering

class PosteriorConfig:
    def __init__(self):
        self.pretrain_model_name_or_path = "bert-base-uncased"

class Posterior(GenerativeBaseModel.GenerativeBaseModel):
    def __init__(self, library, config=PosteriorConfig()):
        super().__init__(library, is_prior=False)
        self.config = config

        bert_config = BertConfig.from_json_file(os.path.join(self.config.pretrain_model_name_or_path, "config.json"))
        self.model = BertForQuestionAnswering(bert_config, 4)
        state_dict = torch.load(os.path.join(self.config.pretrain_model_name_or_path, 'pytorch_model.bin'), map_location='cpu')
        state_dict = {k[5:] if k.startswith("bert.") else k: v for k, v in state_dict.items() if k.startswith('bert.')}
        self.model.bert.load_state_dict(state_dict)

    def set_loss_type(self, loss_type, tau=None):
        self.model.set_loss_type(loss_type, tau=tau)

    def move_to_device(self, device):
        self.to(device)

    def forward(
        self,
        input_ids, attention_mask, token_type_ids,
        start_positions=None, end_positions=None, switch=None, answer_mask=None,
        loss_selection=None,
        global_step=-1
    ):
        device = next(self.parameters()).device
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if attention_mask.device != device:
            attention_mask = attention_mask.to(device)
        if token_type_ids.device != device:
            token_type_ids = token_type_ids.to(device)
        if start_positions is not None and start_positions.device != device:
            start_positions = start_positions.to(device)
        if end_positions is not None and end_positions.device != device:
            end_positions = end_positions.to(device)
        if switch is not None and switch.device != device:
            switch = switch.to(device)
        if answer_mask is not None and answer_mask.device != device:
            answer_mask = answer_mask.to(device)
        if loss_selection is not None and loss_selection.device != device:
            loss_selection = loss_selection.to(device)
        do_predict = any([(item is None) for item in [start_positions, end_positions, switch, answer_mask]])
        if do_predict:
            batch = (input_ids, attention_mask, token_type_ids)
        else:
            batch = (input_ids, attention_mask, token_type_ids, start_positions, end_positions, switch, answer_mask)
        res = self.model(batch, global_step)

        if do_predict:
            return {
                'start_logits': res[0],
                'end_logits': res[1],
                'switch_logits': res[2]
            }
        else:
            loss, loss_tensor = res[0], res[1]
            if loss_selection is not None:
                loss = torch.sum(torch.gather(loss_tensor, 1, loss_selection.unsqueeze(1)))
            return {
                'batch_id2program_log_probs': -loss_tensor,
                'program_generation_loss': loss
            }
