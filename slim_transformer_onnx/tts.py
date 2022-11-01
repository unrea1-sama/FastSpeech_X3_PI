# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
import math

import torch

from .base import BaseModule
from .text_encoder import TextEncoder, MelDecoder
from .length_regulator import LengthRegulator


class SlimTransformerONNX(BaseModule):

    def __init__(self,
                 n_vocab,
                 enc_channels,
                 enc_filter_channels,
                 dp_filter_channels,
                 enc_heads,
                 enc_layers,
                 enc_kernel,
                 enc_dropout,
                 dec_channels,
                 dec_filter_channels,
                 dec_heads,
                 dec_layers,
                 dec_kernel,
                 dec_dropout,
                 mel_dim,enc_max_length,dec_max_length):
        super(SlimTransformerONNX, self).__init__()
        self.n_vocab = n_vocab
        self.enc_channels = enc_channels
        self.enc_filter_channels = enc_filter_channels
        self.dp_filter_channels = dp_filter_channels
        self.enc_heads = enc_heads
        self.enc_layers = enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.dec_channels = dec_channels
        self.dec_filter_channels = dec_filter_channels
        self.dec_heads = dec_heads
        self.dec_layers = dec_layers
        self.dec_kernel = dec_kernel
        self.dec_dropout = dec_dropout
        self.mel_dim = mel_dim
        self.enc_max_length = enc_max_length
        self.dec_max_length = dec_max_length

        self.emb = torch.nn.Embedding(n_vocab, self.enc_channels, padding_idx=0)
        torch.nn.init.normal_(self.emb.weight, 0.0, self.enc_channels**-0.5)

        self.encoder = TextEncoder(n_vocab, self.mel_dim, self.enc_channels,
                                   self.enc_filter_channels, self.dp_filter_channels,
                                   self.enc_heads, self.enc_layers, self.enc_kernel,
                                   self.enc_dropout)
        self.decoder = MelDecoder(self.mel_dim, self.dec_channels,self.dec_filter_channels,
                                  self.dec_heads, self.dec_layers, self.dec_kernel,
                                  self.dec_dropout,self.dec_max_length)
        self.length_regulator = LengthRegulator()
        self.pe_encoder = torch.nn.Parameter(get_pe_table(
            self.enc_max_length, self.enc_channels),
                                             requires_grad=False)
        self.pe_decoder = torch.nn.Parameter(get_pe_table(
            self.dec_max_length, self.dec_channels),
                                             requires_grad=False)

    def forward(self, x, x_lengths, durations=None):
        """
        Args:
            x: shape (b,1,enc_max_length,1)
            x_lengths: shape (b,1,1,1)
            durations: (b,1,1,enc_max_length)
        """
        x = x.long()
        x_lengths = x_lengths.long()
        x = x.squeeze(3)
        x = self.emb(x) * math.sqrt(self.enc_channels)
        # x: (b,1,enc_max_length,d)
        x = x.permute(0, 3, 1, 2)
        # x: (b,d,1,enc_max_length)
        enc_input = x + self.pe_encoder
        # enc_input: (b,d,1,enc_max_length)
        enc_output, logw, x_mask = self.encoder(enc_input, x_lengths)
        # (b,d,1,enc_max_length), (b,1,1,enc_max_length), (b,1,1,enc_max_length)
        if durations is None:
            durations = torch.ceil((torch.exp(logw) - 1)) * x_mask

        dec_input, y_lengths = self.length_regulator(enc_output, durations,
                                                     self.enc_max_length,
                                                     self.dec_max_length)
        # dec_input: (b,d,1,target_max_length)
        # y_lengths: (b,1,1,1)
        dec_input = dec_input + self.pe_decoder
        mel, mel_mask = self.decoder(dec_input, y_lengths)

        return logw, x_mask, mel, mel_mask, y_lengths


def get_pe_table(max_seq_len, input_dim):
    x = torch.arange(max_seq_len).reshape(max_seq_len, 1)
    y = torch.arange(input_dim).reshape(1, input_dim)
    table = x / 10000**(2 *
                        (torch.div(y, 2, rounding_mode='trunc')) / input_dim)
    table[:, 0::2] = torch.sin(table[:, 0::2])  # dim 2i
    table[:, 1::2] = torch.cos(table[:, 1::2])  # dim 2i+1
    # shape (t,d)
    return table.T.reshape(1, input_dim, 1, max_seq_len)
