# -*- coding: utf-8 -*-
import pdb
import torch
import torch.nn as nn
import warnings
from Normal.train import GPU

warnings.filterwarnings('ignore')

device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")


class RetainAttention(nn.Module):

    def __init__(self, embed_size, hidden_size):
        super(RetainAttention, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.attn_beta = nn.Linear(hidden_size, embed_size)
        self.activate_beta = nn.Tanh()
        self.attn_alpha = nn.Linear(hidden_size, 1)
        self.activate_alpha = nn.Softmax(dim=-1)

    def forward(self, data_alpha, data_beta, data_embed):
        # shape of data_alpha: <n_batch, n_seq, hidden_size>
        # shape of data_beta : <n_batch, n_seq, hidden_size>
        # shape of data_embed: <n_batch, n_seq, embed_size>
        # shape of data_mask: <n_batch, n_seq>

        # generate beta weights
        n_batch, n_seq, hidden_size = data_beta.shape
        # shape of beta_weights: <n_batch, n_seq, embed_size>
        beta_weights = self.activate_beta(self.attn_beta(data_beta.reshape(-1, hidden_size)))
        beta_weights = beta_weights.reshape(n_batch, n_seq, self.embed_size)

        # generate alpha weights
        n_batch, n_seq, hidden_size = data_alpha.shape
        # shape of _ori_correlate_value: <n_batch, 1, n_seq>
        _correlate_value = self.attn_alpha(data_alpha.reshape(-1, hidden_size)).reshape(n_batch, n_seq).unsqueeze(1)
        # shape of attention_value_format: <n_batch, 1, n_seq>
        attention_value_format = torch.exp(_correlate_value)
        # shape of mask: <1, n_seq, n_seq>
        #  [[[ 1  0  0
        #      1  1  0
        #      1  1  1 ]]]
        mask = torch.triu(torch.ones([n_seq, n_seq]), diagonal=0).permute(1, 0).unsqueeze(0).to(device)
        # shape of accumulate_attention_value: <n_batch, n_seq, 1>
        accumulate_attention_value = torch.sum(attention_value_format * mask, -1).unsqueeze(-1) + 1e-10
        # shape of each_attention_value: <n_batch, n_seq, n_seq>
        each_attention_value = attention_value_format * mask
        # shape of attention_weight_format: <n_batch, n_seq, n_seq>
        alpha_weights = each_attention_value / accumulate_attention_value

        # shape of _visit_beta_weights: <n_batch, 1, n_seq, embed_size>
        _visit_beta_weights = beta_weights.unsqueeze(1)
        # shape of _visit_alpha_weights: <n_batch, n_seq, n_seq, 1>
        _visit_alpha_weights = alpha_weights.unsqueeze(-1)
        # shape of _visit_data_embed: <n_batch, 1, n_seq, embed_size>
        _visit_data_embed = data_embed.unsqueeze(1)

        # shape of mix_weights: <n_batch, n_seq, n_seq, embed_size>
        mix_weights = _visit_beta_weights * _visit_alpha_weights
        # shape of weighted_output: <n_batch, n_seq, embed_size>
        weighted_output = torch.sum(mix_weights * _visit_data_embed, dim=-2)

        return weighted_output


class RETAIN(nn.Module):
    def __init__(self, icd_size=10, pro_size=10, hidden_size=32, dropout=0.1, batch_first=True):
        super(RETAIN, self).__init__()
        self.dia_embedding = nn.Linear(icd_size, hidden_size)
        self.pro_embedding = nn.Linear(pro_size, hidden_size)
        self.rnn_alpha = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout,
                                bidirectional=False, batch_first=batch_first)
        self.rnn_beta = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout,
                               bidirectional=False, batch_first=batch_first)

        self.attention_func = RetainAttention(hidden_size, hidden_size)
        self.output_func = nn.Linear(hidden_size, icd_size)

    def forward(self, input_data, choice):
        if choice == 'dia':
            embed_data = self.dia_embedding(input_data)
        else:
            embed_data = self.pro_embedding(input_data)
        # embed_data (N, T, H)
        rnn_output_alpha, _ = self.rnn_alpha(embed_data)  # (N, T, H)
        rnn_output_beta, _ = self.rnn_beta(embed_data)  # (N, T, H)
        attn_output = self.attention_func(rnn_output_alpha, rnn_output_beta, embed_data)  # (N, T, H), after mask
        output = self.output_func(attn_output)  # (N, T, U)
        return output

