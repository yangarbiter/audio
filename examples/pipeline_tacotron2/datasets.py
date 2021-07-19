from typing import Tuple

import torch
from torch import Tensor

from torch.utils.data.dataset import random_split
from torchaudio.datasets import LJSPEECH, LIBRITTS

#from text import text_to_sequence
#from text_preprocessing import text_to_sequence


class SpectralNormalization(torch.nn.Module):
    def forward(self, input):
        return torch.log(torch.clamp(input, min=1e-5))


class InverseSpectralNormalization(torch.nn.Module):
    def forward(self, input):
        return torch.exp(input)


class MapMemoryCache(torch.utils.data.Dataset):
    r"""Wrap a dataset so that, whenever a new item is returned, it is saved to memory.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._cache = [None] * len(dataset)

    def __getitem__(self, n):
        if self._cache[n] is not None:
            return self._cache[n]

        item = self.dataset[n]
        self._cache[n] = item

        return item

    def __len__(self):
        return len(self.dataset)


class Processed(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms, text_preprocessor):
        self.dataset = dataset
        self.transforms = transforms
        self.text_preprocessor = text_preprocessor

    def __getitem__(self, key):
        item = self.dataset[key]
        return self.process_datapoint(item)

    def __len__(self):
        return len(self.dataset)

    def process_datapoint(self, item):
        melspec = self.transforms(item[0])
        #text_norm = torch.IntTensor(text_to_sequence(item[2], self.text_cleaners))
        text_norm = torch.IntTensor(self.text_preprocessor(item[2]))
        return text_norm, torch.squeeze(melspec, 0)


def split_process_dataset(dataset, file_path, val_ratio, transforms, text_preprocessor):
    if dataset == 'ljspeech':
        data = LJSPEECH(root=file_path, download=False)

        val_length = int(len(data) * val_ratio)
        lengths = [len(data) - val_length, val_length]
        train_dataset, val_dataset = random_split(data, lengths)

    else:
        raise ValueError(f"Expected dataset: `ljspeech` , but found {dataset}")

    train_dataset = Processed(train_dataset, transforms, text_preprocessor)
    val_dataset = Processed(val_dataset, transforms, text_preprocessor)

    train_dataset = MapMemoryCache(train_dataset)
    val_dataset = MapMemoryCache(val_dataset)

    return train_dataset, val_dataset


def text_mel_collate_fn(batch: Tuple[Tensor, Tensor],
                        n_frames_per_step: int=1) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x[0]) for x in batch]),
        dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]][0]
        text_padded[i, :text.size(0)] = text

    # Right zero-pad mel-spec
    num_mels = batch[0][1].size(0)
    max_target_len = max([x[1].size(1) for x in batch])
    if max_target_len % n_frames_per_step != 0:
        max_target_len += n_frames_per_step - max_target_len % n_frames_per_step
        assert max_target_len % n_frames_per_step == 0

    # include mel padded and gate padded
    mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
    mel_padded.zero_()
    gate_padded = torch.FloatTensor(len(batch), max_target_len)
    gate_padded.zero_()
    output_lengths = torch.LongTensor(len(batch))
    for i in range(len(ids_sorted_decreasing)):
        mel = batch[ids_sorted_decreasing[i]][1]
        mel_padded[i, :, :mel.size(1)] = mel
        gate_padded[i, mel.size(1)-1:] = 1
        output_lengths[i] = mel.size(1)

    return text_padded, input_lengths, mel_padded, gate_padded, output_lengths
