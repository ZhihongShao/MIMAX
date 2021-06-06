import torch
import torch.nn.functional as F

from . import load_pretrain_model
from . import GenerativeBaseModel

class PriorConfig:
    def __init__(self):
        self.pretrain_model_name_or_path = "bart-base"

class Prior(GenerativeBaseModel.GenerativeBaseModel):
    def __init__(self, library, config=PriorConfig()):
        super().__init__(library, is_prior=True)
        self.config = config

        lm_config, self.lm = load_pretrain_model.load_pretrain_lm(
            self.config.pretrain_model_name_or_path
        )
        self.lm.resize_token_embeddings(library.get_num_tokens(is_prior=self.is_prior))

    def move_to_device(self, device):
        self.to(device)

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids
    ):
        '''
        input_len = 1 + question_len + 1 + passage_len + 1 + program_len + 1
        *   Arguments:
            *   input_ids (torch.LongTensor): [batch, max(input_len)]
            *   attention_mask (torch.LongTensor): [batch, max(input_len), max(input_len)]
            *   decoder_input_ids (torch.LongTensor): [batch, max(output_len)]
        *   Returns:
            *   (dict):
                *   loss (torch.FloatTensor): scalar
                *   log_prob (torch.FloatTensor): [batch]
        '''
        device = next(self.parameters()).device
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if attention_mask.device != device:
            attention_mask = attention_mask.to(device)
        if decoder_input_ids.device != device:
            decoder_input_ids = decoder_input_ids.to(device)

        logits = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )[0]

        logits = logits[:, :-1]
        mask = decoder_input_ids.ne(self.library.get_pad_token_id(is_prior=self.is_prior)).float()
        neg_log_prob = torch.sum(F.cross_entropy(logits.permute(0, 2, 1), decoder_input_ids[:, 1:], reduction='none') * mask[:, 1:], 1)
        loss = torch.mean(neg_log_prob)
        log_prob = -neg_log_prob

        return {
            'loss': loss,
            'log_prob': log_prob
        }
