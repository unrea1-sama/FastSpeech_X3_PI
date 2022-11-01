""" from https://github.com/jaywalnut310/glow-tts """

import math

import torch
from torch import nn

from .base import BaseModule
from .utils import sequence_mask

import numpy as np


class LayerNorm(BaseModule):

    def __init__(self, channels, eps=1e-4):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(channels, eps)

    def forward(self, x):
        """
        Args:
            x: shape (b,d,1,t)
        """
        x = x.permute(0, 2, 3, 1)
        # x: (b,1,t,d)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class Conv1d(BaseModule):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels,
                                    out_channels, (1, kernel_size),
                                    padding=(0, kernel_size // 2))
        self.weight = self.conv.weight
        self.bias = self.conv.bias

    def forward(self, x):
        """Using conv2d to perform conv1d.

        Args:
            x: shape (b,d,1,t)
        """
        x = self.conv(x)
        return x


class ConvReluNorm(BaseModule):

    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size,
                 n_layers, p_dropout):
        super(ConvReluNorm, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.conv_layers.append(
            Conv1d(in_channels, hidden_channels, kernel_size))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = torch.nn.Sequential(torch.nn.ReLU(),
                                             torch.nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                Conv1d(hidden_channels, hidden_channels, kernel_size))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        """

        Args:
            x: shape (b,d,1,t)
            x_mask: shape (b,1,1,t)
        """
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DurationPredictor(BaseModule):

    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super(DurationPredictor, self).__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout

        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = Conv1d(in_channels, filter_channels, kernel_size)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = Conv1d(filter_channels, filter_channels, kernel_size)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        """

        Args:
            x: shape (b,d,1,t)
            x_mask: shape (b,1,1,t)
        """
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class MultiHeadAttention(BaseModule):

    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        p_dropout=0.0,
    ):
        super(MultiHeadAttention, self).__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout

        self.k_channels = channels // n_heads
        self.conv_q = Conv1d(channels, channels, 1)
        self.conv_k = Conv1d(channels, channels, 1)
        self.conv_v = Conv1d(channels, channels, 1)

        self.conv_o = Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, max_length, attn_mask):
        # x: (b,d,1,t)
        # c: (b,d,1,t)
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x = self.attention(q, k, v, max_length, attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, max_length, mask):
        query = query.view(
            -1, self.n_heads, self.k_channels, max_length).transpose(
                2, 3)  # (b,heads,channels,t_q)-> (b,heads,t_q,channels)
        key = key.view(-1, self.n_heads, self.k_channels, max_length)
        # (b,heads,channels,t_k)
        value = value.view(-1, self.n_heads, self.k_channels,
                           max_length).transpose(2, 3)
        # (b,heads,channels,t_k)-> (b,heads,t_k,channels)

        scores = torch.matmul(query, key) / math.sqrt(self.k_channels)
        # scores: (b,heads,t_q,t_k)
        #if mask is not None:
        #scores = scores.masked_fill(mask == 0, -1e4)
        scores = scores * mask + (1 - mask) * -1e5
        p_attn = torch.nn.functional.softmax(scores, dim=3)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        # (b,heads,t_q,t_k)*(b,heads,t_k,channels)->(b,heads,t_q,channels)

        output = output.transpose(2, 3).reshape(-1, self.channels, 1,
                                                max_length)
        return output


class FFN(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 filter_channels,
                 kernel_size,
                 p_dropout=0.0):
        super(FFN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = torch.nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class Encoder(BaseModule):

    def __init__(self,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size=1,
                 p_dropout=0.0):
        super(Encoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = torch.nn.Dropout(p_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(hidden_channels,
                                   hidden_channels,
                                   n_heads,
                                   p_dropout=p_dropout))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, max_length):
        """

        Args:
            x (_type_): (b,d,1,t)
            x_mask (_type_): (b,1,1,t)
        """
        attn_mask = x_mask.permute(0, 1, 3, 2) * x_mask
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, max_length, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class TextEncoder(BaseModule):

    def __init__(self,
                 n_vocab,
                 n_feats,
                 n_channels,
                 filter_channels,
                 filter_channels_dp,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 window_size=None,
                 spk_emb_dim=64,
                 n_spks=1):
        super(TextEncoder, self).__init__()
        self.n_vocab = n_vocab
        self.n_feats = n_feats
        self.n_channels = n_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.spk_emb_dim = spk_emb_dim
        self.n_spks = n_spks

        self.prenet = ConvReluNorm(n_channels,
                                   n_channels,
                                   n_channels,
                                   kernel_size=5,
                                   n_layers=3,
                                   p_dropout=0.5)

        self.encoder = Encoder(n_channels + (spk_emb_dim if n_spks > 1 else 0),
                               filter_channels, n_heads, n_layers, kernel_size,
                               0)

        self.proj_m = Conv1d(n_channels + (spk_emb_dim if n_spks > 1 else 0),
                             n_channels, 1)
        self.proj_w = DurationPredictor(
            n_channels + (spk_emb_dim if n_spks > 1 else 0),
            filter_channels_dp, 5, 0.2)

        self.enc_max_length = 128

    def forward(self, x, x_lengths):
        # x: (b,d,1,enc_max_length)
        # x_length: (b,1,1,1)
        x_mask = sequence_mask(x_lengths, self.enc_max_length)
        # x_mask: (b,1,1,enc_max_length)
        x = self.prenet(x, x_mask)

        x = self.encoder(x, x_mask, self.enc_max_length)
        # x: (b,n_channels,1,enc_max_length)

        logw = self.proj_w(x, x_mask)
        # logw: (b,1,1,enc_max_length)

        x = self.proj_m(x) * x_mask
        return x, logw, x_mask


class MelDecoder(BaseModule):

    def __init__(self, n_feats, n_channels, filter_channels, n_heads, n_layers,
                 kernel_size, p_dropout,dec_max_length):
        super(MelDecoder, self).__init__()
        self.n_feats = n_feats
        self.n_channels = n_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.prenet = ConvReluNorm(n_channels,
                                   n_channels,
                                   n_channels,
                                   kernel_size=5,
                                   n_layers=3,
                                   p_dropout=0.5)

        self.encoder = Encoder(n_channels, filter_channels, n_heads, n_layers,
                               kernel_size, 0)
        self.linear = torch.nn.Linear(n_channels, n_feats)
        self.proj = Conv1d(n_channels, n_feats, 1)
        self.dec_max_length = dec_max_length

    def forward(self, x, x_lengths):
        """
        Args:
            x: shape (b,d,1,dec_max_length)
            x_lengths: shape (b,1,1,1)

        """
        x_mask = sequence_mask(x_lengths, self.dec_max_length)
        # x_mask: (b,1,1,dec_max_length)
        x = self.prenet(x, x_mask)
        mel = self.encoder(x, x_mask, self.dec_max_length)
        # x: (b,n_channels,1,dec_max_length)
        mel = self.proj(mel) * x_mask
        # x: (b,n_feats,1,dec_max_length)
        return mel, x_mask
