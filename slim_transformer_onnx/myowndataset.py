import torch
import torch.utils.data
import json
import torchaudio

from torch.nn.utils.rnn import pad_sequence
from .feature import LogMelFBank


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset_path, phnset_path, sample_rate, window_size, hop_size, mel_dim
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_dim = mel_dim
        self.mel_extractor = LogMelFBank(
            sample_rate, window_size, hop_size, window_size, mel_dim
        )
        with open(dataset_path) as f:
            self.samples = json.load(f)
        with open(phnset_path) as f:
            # 0 is reserved for PAD token
            self.phnset = {x: i+1 for i, x in enumerate(json.load(f))}

    def get_vocab_size(self):
        # PAD is also considered
        return len(self.phnset) + 1

    def _load_mel(self, path):
        audio, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate).clamp(
                min=-1, max=1
            )
            # audio: (1,t)
        mel = self.mel_extractor.get_mel_spectrogram(audio[0]).T
        # mel: (t,d)
        return mel, audio

    def _load_text(self, phonemes):
        durations = []
        phn_ids = []
        phns = []
        for phoneme in phonemes:
            phn, start, end, duration = phoneme
            phn_ids.append(self.phnset[phn])
            duration = (
                int(end * self.sample_rate // self.hop_size)
                - int(start * self.sample_rate // self.hop_size)
            )
            durations.append(duration)
            phns.append(phn)
        return phn_ids, durations, phns

    def __getitem__(self, i):
        phn_ids, durations, phns = self._load_text(self.samples[i]["phonemes"])
        mel, audio = self._load_mel(self.samples[i]["path"])
        mel = mel[: sum(durations)]
        return {
            "phn_ids": phn_ids,
            "durations": durations,
            "mel": mel,
            "wav": audio,
            "phns": phns,
            "item_name": self.samples[i]["item_name"],
        }

    def __len__(self):
        return len(self.samples)


class collateFn:
    def __init__(self, x_max_length=None, y_max_length=None):
        self.x_max_length = x_max_length
        self.y_max_length = y_max_length

    def __call__(self, batch):
        phs_lengths, sorted_idx = torch.sort(
            torch.LongTensor([len(x["phn_ids"]) for x in batch]), descending=True
        )

        mel_lengths = torch.tensor([batch[i]["mel"].shape[0] for i in sorted_idx])
        padded_phs = pad_sequence(
            [
                torch.tensor(pad_list(batch[i]["phn_ids"], self.x_max_length))
                for i in sorted_idx
            ],
            batch_first=True,
        )

        padded_durations = pad_sequence(
            [
                torch.tensor(pad_list(batch[i]["durations"], self.x_max_length))
                for i in sorted_idx
            ],
            batch_first=True,
        )

        padded_mels = pad_sequence(
            [batch[i]["mel"] for i in sorted_idx], batch_first=True
        )
        if self.y_max_length is not None:
            padded_mels = torch.nn.functional.pad(
                padded_mels, (0, 0, 0, max(self.y_max_length - padded_mels.shape[1], 0))
            )
        txts = [batch[i]["phns"] for i in sorted_idx]
        wavs = [batch[i]["wav"] for i in sorted_idx]
        item_names = [batch[i]["item_name"] for i in sorted_idx]

        return {
            "x": padded_phs,
            "x_lengths": phs_lengths,
            "y": padded_mels.permute(0, 2, 1),
            "y_lengths": mel_lengths,
            "duration": padded_durations,
            "phns": txts,
            "wavs": wavs,
            "names": item_names,
        }


def pad_list(l, length=None):
    if length is None:
        return l
    if len(l) > length:
        return l[:length]
    elif len(l) < length:
        return l + [0] * (length - len(l))
    return l
