# -*- coding: utf-8 -*-
import pdb
import torch
import torch.nn as nn
import warnings
from Stable.train import GPU

warnings.filterwarnings('ignore')

device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")


class LocationAttention(nn.Module):

    def __init__(self, hidden_size):
        super(LocationAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_value_ori_func = nn.Linear(self.hidden_size, 1)

    def forward(self, input_data):
        # shape of input_data: <n_batch, n_seq, hidden_size>
        n_batch, n_seq, hidden_size = input_data.shape
        # shape of reshape_feat: <n_batch*n_seq, hidden_size>
        reshape_feat = input_data.reshape(n_batch * n_seq, hidden_size)
        # shape of attention_value_ori: <n_batch*n_seq, 1>
        attention_value_ori = torch.exp(self.attention_value_ori_func(reshape_feat))
        # shape of attention_value_format: <n_batch, 1, n_seq>
        attention_value_format = attention_value_ori.reshape(n_batch, n_seq).unsqueeze(1)
        # shape of ensemble flag format: <1, n_seq, n_seq>
        # if n_seq = 3, ensemble_flag_format can get below flag data
        #  [[[ 1  0  0
        #      1  1  0
        #      1  1  1 ]]]
        ensemble_flag_format = torch.triu(torch.ones([n_seq, n_seq]), diagonal=0).permute(1, 0).unsqueeze(0).to(device)
        # shape of accumulate_attention_value: <n_batch, n_seq, 1>
        accumulate_attention_value = torch.sum(attention_value_format * ensemble_flag_format, -1).unsqueeze(-1) + 1e-9
        # shape of each_attention_value: <n_batch, n_seq, n_seq>
        each_attention_value = attention_value_format * ensemble_flag_format
        # shape of attention_weight_format: <n_batch, n_seq, n_seq>
        attention_weight_format = each_attention_value / accumulate_attention_value
        # shape of _extend_attention_weight_format: <n_batch, n_seq, n_seq, 1>
        _extend_attention_weight_format = attention_weight_format.unsqueeze(-1)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>
        _extend_input_data = input_data.unsqueeze(1)
        # shape of _weighted_input_data: <n_batch, n_seq, n_seq, hidden_size>
        _weighted_input_data = _extend_attention_weight_format * _extend_input_data
        # shape of weighted_output: <n_batch, n_seq, hidden_size>
        weighted_output = torch.sum(_weighted_input_data, 2)
        return weighted_output


class GeneralAttention(nn.Module):

    def __init__(self, hidden_size):
        super(GeneralAttention, self).__init__()
        self.hidden_size = hidden_size
        self.correlated_value_ori_func = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input_data):
        # shape of input_data: <n_batch, n_seq, hidden_size>
        n_batch, n_seq, hidden_size = input_data.shape
        # shape of reshape_feat: <n_batch*n_seq, hidden_size>
        reshape_feat = input_data.reshape(n_batch * n_seq, hidden_size)
        # shape of correlated_value_ori: <n_batch, n_seq, hidden_size>
        correlated_value_ori = self.correlated_value_ori_func(reshape_feat).reshape(n_batch, n_seq, hidden_size)
        # shape of _extend_correlated_value_ori: <n_batch, n_seq, 1, hidden_size>
        _extend_correlated_value_ori = correlated_value_ori.unsqueeze(-2)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>
        _extend_input_data = input_data.unsqueeze(1)
        # shape of _extend_input_data: <n_batch, n_seq, n_seq, hidden_size>
        _correlat_value = _extend_correlated_value_ori * _extend_input_data
        # shape of attention_value_format: <n_batch, n_seq, n_seq>
        attention_value_format = torch.exp(torch.sum(_correlat_value, dim=-1))
        # shape of ensemble flag format: <1, n_seq, n_seq>
        # if n_seq = 3, ensemble_flag_format can get below flag data
        #  [[[ 1  0  0
        #      1  1  0
        #      1  1  1 ]]]
        ensemble_flag_format = torch.triu(torch.ones([n_seq, n_seq]), diagonal=0).permute(1, 0).unsqueeze(0).to(device)
        # shape of accumulate_attention_value: <n_batch, n_seq, 1>
        accumulate_attention_value = torch.sum(attention_value_format * ensemble_flag_format, -1).unsqueeze(-1) + 1e-10
        # shape of each_attention_value: <n_batch, n_seq, n_seq>
        each_attention_value = attention_value_format * ensemble_flag_format
        # shape of attention_weight_format: <n_batch, n_seq, n_seq>
        attention_weight_format = each_attention_value / accumulate_attention_value
        # shape of _extend_attention_weight_format: <n_batch, n_seq, n_seq, 1>
        _extend_attention_weight_format = attention_weight_format.unsqueeze(-1)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>
        _extend_input_data = input_data.unsqueeze(1)
        # shape of _weighted_input_data: <n_batch, n_seq, n_seq, hidden_size>
        _weighted_input_data = _extend_attention_weight_format * _extend_input_data
        # shape of weighted_output: <n_batch, n_seq, hidden_size>
        weighted_output = torch.sum(_weighted_input_data, 2)
        return weighted_output


class ConcatenationAttention(nn.Module):
    def __init__(self, hidden_size, attention_dim=16):
        super(ConcatenationAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        self.attention_map_func = nn.Linear(2 * self.hidden_size, self.attention_dim)
        self.activate_func = nn.Tanh()
        self.correlated_value_ori_func = nn.Linear(self.attention_dim, 1)

    def forward(self, input_data):
        # shape of input_data: <n_batch, n_seq, hidden_size>
        n_batch, n_seq, hidden_size = input_data.shape
        # shape of _extend_input_data: <n_batch, n_seq, 1, hidden_size>
        _extend_input_data_f = input_data.unsqueeze(-2)
        # shape of _repeat_extend_correlated_value_ori: <n_batch, n_seq, n_seq, hidden_size>
        _repeat_extend_input_data_f = _extend_input_data_f.repeat(1, 1, n_seq, 1)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>
        _extend_input_data_b = input_data.unsqueeze(1)
        # shape of _repeat_extend_input_data: <n_batch, n_seq, n_seq, hidden_size>
        _repeat_extend_input_data_b = _extend_input_data_b.repeat(1, n_seq, 1, 1)
        # shape of _concate_value: <n_batch, n_seq, n_seq, 2 * hidden_size>
        _concate_value = torch.cat([_repeat_extend_input_data_f, _repeat_extend_input_data_b], dim=-1)
        # shape of _correlat_value: <n_batch, n_seq, n_seq>
        _correlat_value = self.activate_func(self.attention_map_func(_concate_value.reshape(-1, 2 * hidden_size)))
        _correlat_value = self.correlated_value_ori_func(_correlat_value).reshape(n_batch, n_seq, n_seq)
        # shape of attention_value_format: <n_batch, n_seq, n_seq>
        attention_value_format = torch.exp(_correlat_value)
        # shape of ensemble flag format: <1, n_seq, n_seq>
        # if n_seq = 3, ensemble_flag_format can get below flag data
        #  [[[ 1  0  0
        #      1  1  0
        #      1  1  1 ]]]
        ensemble_flag_format = torch.triu(torch.ones([n_seq, n_seq]), diagonal=0).permute(1, 0).unsqueeze(0).to(device)
        # shape of accumulate_attention_value: <n_batch, n_seq, 1>
        accumulate_attention_value = torch.sum(attention_value_format * ensemble_flag_format, -1).unsqueeze(-1) + 1e-10
        # shape of each_attention_value: <n_batch, n_seq, n_seq>
        each_attention_value = attention_value_format * ensemble_flag_format
        # shape of attention_weight_format: <n_batch, n_seq, n_seq>
        attention_weight_format = each_attention_value / accumulate_attention_value
        # shape of _extend_attention_weight_format: <n_batch, n_seq, n_seq, 1>
        _extend_attention_weight_format = attention_weight_format.unsqueeze(-1)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>
        _extend_input_data = input_data.unsqueeze(1)
        # shape of _weighted_input_data: <n_batch, n_seq, n_seq, hidden_size>
        _weighted_input_data = _extend_attention_weight_format * _extend_input_data
        # shape of weighted_output: <n_batch, n_seq, hidden_size>
        weighted_output = torch.sum(_weighted_input_data, 2)
        return weighted_output


class Dipole(nn.Module):
    def __init__(self, attention_type='location_based', icd_size=10, pro_size=10, attention_dim=16,
                 hidden_size=16, dropout=0.1, batch_first=True):
        """
        attention_type : str, optional (default = 'location_based')
            Apply attention mechnism to derive a context vector that captures relevant information to
            help predict target.
            Current support attention methods in [location_based, general, concatenation_based] proposed in KDD2017
            'location_based'      ---> Location-based Attention. A location-based attention function is to
                                       calculate the weights solely from hidden state
            'general'             ---> General Attention. An easy way to capture the relationship between two hidden states
            'concatenation_based' ---> Concatenation-based Attention. Via concatenating two hidden states, then use multi-layer
                                       perceptron(MLP) to calculate the context vector
        """

        super(Dipole, self).__init__()
        self.dia_embedding = nn.Linear(icd_size, hidden_size)
        self.pro_embedding = nn.Linear(pro_size, hidden_size)
        self.rnn_model = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout,
                                bidirectional=True, batch_first=batch_first)
        if attention_type == 'location_based':
            self.attention_func = LocationAttention(2*hidden_size)
        elif attention_type == 'general':
            self.attention_func = GeneralAttention(2*hidden_size)
        elif attention_type == 'concatenation_based':
            self.attention_func = ConcatenationAttention(2*hidden_size, attention_dim)
        else:
            raise Exception('fill in correct attention_type, [location_based, general, concatenation_based]')
        self.output_func = nn.Linear(4*hidden_size, icd_size)

    def forward(self, input_data, choice):
        if choice == 'dia':
            embed_data = self.dia_embedding(input_data)
        else:
            embed_data = self.pro_embedding(input_data)
        # embed_data (N, T, H)
        rnn_output, _ = self.rnn_model(embed_data)  # (N, T, 2H)
        attention_output = self.attention_func(rnn_output)  # (N, T, 2H), after mask
        mix_output = torch.cat([rnn_output, attention_output], dim=-1)  # (N, T, 4H)
        output = self.output_func(mix_output)  # (N, T, U)
        return output, mix_output

