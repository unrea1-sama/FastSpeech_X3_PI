import torch

from .base import BaseModule
from .utils import sequence_mask


def get_padding_mask(lengths, max_len=None):
    """Generate padding mask according to length of the input.

    Args:
        lengths:
            A tensor of shape (b), where b is the batch size.

    Return:
        A mask tensor of shape (b,max_seq_len), where max_seq_len is the length
        of the longest sequence. Positions of padded elements will be set to
        True.
    """
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).int()

    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(
        batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def get_content_mask(length, max_len=None):
    """Generate content mask according to length of the input.

    Args:
        lengths:
            A tensor of shape (b), where b is the batch size.

    Return:
        A mask tensor of shape (b,max_seq_len), where max_seq_len is the length
        of the longest sequence. Positions of padded elements will be set to
        False.
    """
    return ~get_padding_mask(length, max_len)


class LengthRegulator(BaseModule):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def forward(self, x, repeat_count, source_max_length, target_max_length):
        """Repeating all phonemes according to its duration.

        Args:
            x (torch.Tensor): Input phoneme sequences of shape (b,d,1,source_max_length)
            repeat_count (torch.Tensor): Duration of each phoneme of shape
            (b,1,1,source_max_length)
        """
        #repeat_count = repeat_count.long()

        cum_duration = torch.cumsum(repeat_count, dim=3)  # (b,1,1,t_x)
        # output_max_seq_len = torch.max(cum_duration)

        M = sequence_mask(cum_duration.flatten().reshape(-1, 1, 1, 1),
                          target_max_length)
        # M: (b*source_max_length,1,1,target_max_length)
        M = M.reshape(-1, 1, source_max_length, target_max_length)
        # (b,1,source_max_length,target_max_length)
        M[:, :, 1:, :] = M[:, :, 1:, :] - M[:, :, :-1, :]
        # (b,1,source_max_length,target_max_length)
        # (b,d,1,source_max_length)*(b,1,source_max_length,target_max_length)->(b,d,1,target_max_length)
        y = torch.matmul(x, M)
        return y, torch.max(cum_duration, dim=3, keepdim=True)[0]
