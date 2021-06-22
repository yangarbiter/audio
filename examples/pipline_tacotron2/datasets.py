import os
import random

import numpy as np
from scipy.io.wavfile import read
import torch
from torch.utils.data.dataset import random_split
from torchaudio.datasets import LJSPEECH, LIBRITTS

from text import text_to_sequence
from common.stft import STFT
from common.layers import TacotronSTFT


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
    def __init__(self, dataset, transforms, text_cleaners=['english_cleaners']):
        self.dataset = dataset
        self.transforms = transforms
        self.text_cleaners = text_cleaners

    def __getitem__(self, key):
        item = self.dataset[key]
        return self.process_datapoint(item)

    def __len__(self):
        return len(self.dataset)

    def process_datapoint(self, item):
        melspec = self.transforms(item[0])
        text_norm = torch.IntTensor(text_to_sequence(item[2], self.text_cleaners))
        return (text_norm, torch.squeeze(melspec, 0), len(text_norm))


def split_process_dataset(dataset, file_path, val_ratio, transforms):
    if dataset == 'ljspeech':
        data = LJSPEECH(root=file_path, download=False)

        val_length = int(len(data) * val_ratio)
        lengths = [len(data) - val_length, val_length]
        train_dataset, val_dataset = random_split(data, lengths)

    elif dataset == 'libritts':
        train_dataset = LIBRITTS(root=file_path, url='train-clean-100', download=False)
        val_dataset = LIBRITTS(root=file_path, url='dev-clean', download=False)

    else:
        raise ValueError(f"Expected dataset: `ljspeech` or `libritts`, but found {dataset}")

    train_dataset = Processed(train_dataset, transforms)
    val_dataset = Processed(val_dataset, transforms)

    train_dataset = MapMemoryCache(train_dataset)
    val_dataset = MapMemoryCache(val_dataset)

    return train_dataset, val_dataset


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(dataset_path, filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        def split_line(root, line):
            parts = line.strip().split(split)
            if len(parts) > 2:
                raise Exception(
                    "incorrect line format for file: {}".format(filename))
            path = os.path.join(root, parts[0])
            text = parts[1]
            return path,text
        filepaths_and_text = [split_line(dataset_path, line) for line in f]
    return filepaths_and_text


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, dataset_path, audiopaths_and_text, args):
        self.audiopaths_and_text = load_filepaths_and_text(dataset_path, audiopaths_and_text)
        self.text_cleaners = args.text_cleaners
        self.max_wav_value = args.max_wav_value
        self.sample_rate = args.sample_rate
        self.load_mel_from_disk = args.load_mel_from_disk
        self.stft = TacotronSTFT(
            args.n_fft, args.hop_length, args.win_length,
            args.n_mels, args.sample_rate, args.mel_fmin,
            args.mel_fmax)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        len_text = len(text)
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel, len_text)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.load(filename)
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
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
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

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

        # count number of items - characters in text
        #len_x = [x[2] for x in batch]
        #len_x = torch.Tensor(len_x)

        # Return any extra fields as sorted lists
        num_fields = len(batch[0])
        extra_fields = tuple([batch[i][f] for i in ids_sorted_decreasing]
                             for f in range(3, num_fields))

        return (text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths) + extra_fields