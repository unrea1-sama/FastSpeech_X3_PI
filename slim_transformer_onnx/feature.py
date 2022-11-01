# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#               2022 Tsinghua University (Jie Chen)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from espnet(https://github.com/espnet/espnet) and
# wetts(https://github.com/wenet-e2e/wetts)
import librosa
import torch
import torchaudio


class LogMelFBank:
    def __init__(
        self,
        sr,
        n_fft,
        hop_length,
        win_length,
        n_mels,
        fmin=0,
        fmax=None,
    ):
        """Melspectrogram extractor.
        Args:
            sr (int): sampling rate of the incoming signal
            n_fft (int): number of FFT components
            hop_length (int):  the distance between neighboring sliding window frames
            win_length (int): the size of window frame and STFT filter
            n_mels (int): number of Mel bands to generate
            fmin (int): lowest frequency (in Hz)
            fmax (int): highest frequency (in Hz)
        """
        super().__init__()
        self.sr = sr
        # stft
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        # mel
        self.n_mels = n_mels

        self.window = torch.hann_window(win_length)
        self.mel_filter = torch.from_numpy(
            librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        )

    def get_mel_spectrogram(self, wav):
        return torch_melspectrogram(
            wav,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
            self.mel_filter,
        )

    def get_linear_spectrogram(self, wav):
        return torch_linear_spectrogram(
            wav, self.n_fft, self.hop_length, self.win_length, self.window
        )


def torch_stft(x, n_fft, hop_length, win_length, window):
    """Performing STFT using torch.
    Args:
        x (torch.Tensor): input signal. Shape (*,t). * is an optional batch
        dimension.
        n_fft (int): size of Fourier transform.
        hop_length (int): the distance between neighboring sliding window
        frames.
        win_length (int): the size of window frame and STFT filter.
        window (torch.Tensor): window tensor. Shape (win_length).
    Returns: STFT of shape (*,1+n_fft/2,t)
    """
    return torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        onesided=True,
        return_complex=True,
    )


def torch_melspectrogram(
    x, n_fft, hop_length, win_length, window, mel_basis, min_amp=1e-5
):
    """Calculating melspectrogram using torch.
    Args:
        x (torch.Tensor): input signal. Shape (*,t). * is an optional batch
        dimension.
        n_fft (int): size of Fourier transform.
        hop_length (int): the distance between neighboring sliding window
        frames.
        win_length (int): the size of window frame and STFT filter.
        window (torch.Tensor): window tensor. Shape (win_length).
        mel_basis (torch.Tensor): mel filter-bank of shape (n_mels,1+n_fft/2).
        min_amp (float): minimum amplitude. Defaults to 1e-5.
    Returns: melspectrogram of shape (*,n_mels,t)
    """
    stft = torch_stft(x, n_fft, hop_length, win_length, window)
    spec = torch.matmul(mel_basis, torch.abs(stft))
    return torch.log10(torch.clamp(torch.abs(spec), min=min_amp))


def torch_linear_spectrogram(x, n_fft, hop_length, win_length, window, min_amp=1e-5):
    """Calculating linear spectrogram using torch.
    Args:
        x (torch.Tensor): input signal. Shape (*,t). * is an optional batch
        dimension.
        n_fft (int): size of Fourier transform.
        hop_length (int): the distance between neighboring sliding window
        frames.
        win_length (int): the size of window frame and STFT filter.
        window (torch.Tensor): window tensor. Shape (win_length).
        min_amp (float): minimum amplitude. Defaults to 1e-5.
    Returns: spectrogram of shape (*,1+n_fft/2,t)
    """
    stft = torch_stft(x, n_fft, hop_length, win_length, window)
    return torch.log10(torch.clamp(torch.abs(stft), min=min_amp))