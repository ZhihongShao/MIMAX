import torch

from . import load_pretrain_model
from GenerativeModels import GenerativeBaseModel

from GenerativeModels.sqlova.model.nl2sql import wikisql_models

class PosteriorConfig:
    def __init__(self):
        self.num_target_layers = 2
        self.num_lstm_layers = 2
        self.lstm_hidden_dim = 100
        self.dropout = 0.3
        self.pretrain_model_name_or_path = "bert-base-uncased"

class Posterior(GenerativeBaseModel.GenerativeBaseModel):
    def __init__(self, library, config=PosteriorConfig()):
        super().__init__(library, is_prior=False)
        self.config = config

        self.enc_config, self.encoder = load_pretrain_model.load_pretrain_encoder(
            self.config.pretrain_model_name_or_path, use_model_for_classification=False,
            output_hidden_states=True
        )

        self.program_decoder = wikisql_models.Seq2SQL_v1(
            self.enc_config.hidden_size * self.config.num_target_layers,
            self.config.lstm_hidden_dim,
            self.config.num_lstm_layers,
            self.config.dropout,
            library.get_num_cond_ops(),
            library.get_num_agg_ops()
        )

    def move_to_device(self, device):
        self.to(device)

    def forward(
        self,
        input_ids,
        input_mask,
        segment_ids,

        question_indices,
        header_indices,

        num_question_tokens,
        num_header_tokens,
        num_headers,

        select_col=None,
        agg_op=None,
        num_conds=None,
        where_cols=None,
        where_ops=None,
        where_values=None,
    ):
        '''
        *   Arguments:
            *   input_ids (torch.LongTensor): [batch, max(1 + question_len + 1 + headers_len + num_headers)]
            *   input_mask (torch.FloatTensor): [batch, max(1 + question_len + 1 + headers_len + num_headers)]
            *   segment_ids (torch.LongTensor): [batch, max(1 + question_len + 1 + headers_len + num_headers)]
            *   question_indices (torch.LongTensor): [batch, max_question_len], -1 for padding
            *   headers_indices (torch.LongTensor): [sum(num_headers), max_header_len], -1 for padding
            *   num_question_tokens (torch.LongTensor): [batch]
            *   num_header_tokens (torch.LongTensor): [sum(num_headers)]
            *   num_headers (torch.LongTensor): [batch]
            *   select_col (torch.LongTensor): [batch]
            *   agg_op (torch.LongTensor): [batch]
            *   num_conds (torch.LongTensor): [batch]
            *   where_cols (torch.LongTensor): [batch, max_num_cond]
            *   where_ops (torch.LongTensor): [batch, max_num_cond]
            *   where_values (torch.LongTensor): [batch, max_num_cond, 2]
        *   Returns:
            *   if not do_predict:
                *   cross entropy loss (torch.scalar)
                    *   program_generation_loss
                *   batch_id2program_log_probs (torch.FloatTensor): [batch]
            *   if do_predict:
                *   batch_id2logit_select_col (torch.FloatTensor): [batch, num_cols]
                *   batch_id2logit_agg_op (torch.FloatTensor): [batch, num_ops]
                *   batch_id2logit_num_conds (torch.FloatTensor): [batch, max_num_conds]
                *   batch_id2logits_where_cols (torch.FloatTensor): [batch, max_num_conds, num_cols]
                *   batch_id2logits_where_ops (torch.FloatTensor): [batch, max_num_conds, num_ops]
                *   batch_id2logits_where_values (torch.FloatTensor): [batch, max_num_conds, question_len, 2]
        '''
        device = next(self.parameters()).device
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if input_mask.device != device:
            input_mask = input_mask.to(device)
        if segment_ids.device != device:
            segment_ids = segment_ids.to(device)
        if question_indices.device != device:
            question_indices = question_indices.to(device)
        if header_indices.device != device:
            header_indices = header_indices.to(device)
        if select_col is not None and select_col.device != device:
            select_col = select_col.to(device)
        if agg_op is not None and agg_op.device != device:
            agg_op = agg_op.to(device)
        if where_cols is not None and where_cols.device != device:
            where_cols = where_cols.to(device)
        if where_ops is not None and where_ops.device != device:
            where_ops = where_ops.to(device)
        if where_values is not None and where_values.device != device:
            where_values = where_values.to(device)
        do_predict = any(inp is None for inp in [select_col, agg_op, num_conds, where_cols, where_ops, where_values])

        all_encoder_layer = self.encoder(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[2]
        enc = torch.cat(list(reversed(all_encoder_layer[-self.config.num_target_layers:])), 2)

        zero_padded_enc = torch.cat((
            torch.zeros((enc.size(0), 1, enc.size(2)), device=device, dtype=torch.float32),
            enc
        ), 1)
        enc_question = torch.gather(zero_padded_enc, 1, question_indices.unsqueeze(2).expand(-1, -1, enc.size(2)) + 1)
        
        tot_num_headers, max_header_len = header_indices.size()[0], header_indices.size()[1]
        zero_padded_flag_enc = torch.cat((
            torch.zeros((1, enc.size(2)), device=device, dtype=torch.float32),
            enc.contiguous().view(enc.size(0) * enc.size(1), enc.size(2))
        ), 0)
        enc_header = torch.gather(zero_padded_flag_enc, 0, (header_indices.contiguous().view(tot_num_headers * max_header_len, 1) + 1).expand(-1, enc.size(2))).contiguous().view(tot_num_headers, max_header_len, enc.size(2))

        num_question_tokens_lst = num_question_tokens.cpu().numpy().tolist()
        num_header_tokens_lst = num_header_tokens.cpu().numpy().tolist()
        num_headers_lst = num_headers.cpu().numpy().tolist()

        if do_predict:
            logit_select_col, logit_agg_op, logit_num_conds, logits_where_cols, logits_where_ops, logits_where_values = self.program_decoder(enc_question, num_question_tokens_lst, enc_header, num_header_tokens_lst, num_headers_lst)
            return {
                'batch_id2logit_select_col': logit_select_col,
                'batch_id2logit_agg_op': logit_agg_op,
                'batch_id2logit_num_conds': logit_num_conds,
                'batch_id2logits_where_cols': logits_where_cols,
                'batch_id2logits_where_ops': logits_where_ops,
                'batch_id2logits_where_values': logits_where_values
            }
        else:
            select_col_lst = select_col.cpu().numpy().tolist()
            agg_op_lst = agg_op.cpu().numpy().tolist()
            num_conds_lst = num_conds.cpu().numpy().tolist()
            where_cols_lst = [wc[:n] for wc, n in zip(where_cols.cpu().numpy().tolist(), num_conds_lst)]
            where_ops_lst = [wo[:n] for wo, n in zip(where_ops.cpu().numpy().tolist(), num_conds_lst)]
            where_values_lst = [wvi[:n] for wvi, n in zip(where_values.cpu().numpy().tolist(), num_conds_lst)]

            logit_select_col, logit_agg_op, logit_num_conds, logits_where_cols, logits_where_ops, logits_where_values = self.program_decoder(enc_question, num_question_tokens_lst, enc_header, num_header_tokens_lst, num_headers_lst, g_sc=select_col_lst, g_sa=agg_op_lst, g_wn=num_conds_lst, g_wc=where_cols_lst, g_wvi=where_values_lst)

            batch_id2neg_log_prob = wikisql_models.Loss_sw_se(logit_select_col, logit_agg_op, logit_num_conds, logits_where_cols, logits_where_ops, logits_where_values, \
                select_col_lst, agg_op_lst, num_conds_lst, where_cols_lst, where_ops_lst, where_values_lst, reduction='none')
            batch_id2log_prob = -batch_id2neg_log_prob
            return {
                'batch_id2program_log_probs': batch_id2log_prob,
                'program_generation_loss': torch.sum(batch_id2neg_log_prob)
            }
