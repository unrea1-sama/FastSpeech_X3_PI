""" from https://github.com/jaywalnut310/glow-tts """

import torch
import librosa
import numpy as np


def sequence_mask(length, max_length=None):
    """Generating mask tensor according to `length`.

    Args:
        length (Tensor): length of shape (b,1,1,1).
        max_length (int, optional): max length. Defaults to None.

    Returns:
        Tensor: mask tensor of shape (b,t), where t is the maximum of `length`.
        True indicates a non-padding element.
    """

    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=torch.float32,
                     device=length.device).reshape(1, 1, 1, -1)
    # return shape: (b,1,1,max_length)
    return (x < length).float()


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(
        path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


def duration_loss(logw, logw_, lengths):
    loss = torch.sum((logw - logw_)**2) / torch.sum(lengths)
    return loss


def get_mcd(ground_truth_mel, predicted_mel):
    """Getting MCD from dtw.

    Args:
        ground_truth_mel: Ground truth mel. Shape (mel_d,t1)
        predicted_mel: Predicted mel. Shape (mel_d,t2)
    """
    cost = librosa.sequence.dtw(ground_truth_mel,
                                predicted_mel,
                                backtrack=False)
    return cost[-1, -1] / ground_truth_mel.shape[1]
