# -*- coding: utf-8 -*-
import pdb
import torch
import torch.nn as nn
import warnings
from Normal.train import GPU

warnings.filterwarnings('ignore')

device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")


class LSTM(nn.Module):
    def __init__(self, icd_size=10, pro_size=10, hidden_size=16, dropout=0.1, batch_first=True):
        super(LSTM, self).__init__()
        self.dia_embedding = nn.Linear(icd_size, hidden_size)
        self.pro_embedding = nn.Linear(pro_size, hidden_size)
        self.rnn_model = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout,
                                 bidirectional=False, batch_first=batch_first)
        self.output_func = nn.Linear(hidden_size, icd_size)

    def forward(self, input_data, choice):
        if choice == 'dia':
            embed_data = self.dia_embedding(input_data)
        else:
            embed_data = self.pro_embedding(input_data)

        # embed_data (N, T, H)
        rnn_output, _ = self.rnn_model(embed_data)  # (N, T, H)
        output = self.output_func(rnn_output)  # (N, T, U)
        pdb.set_trace()
        return output

