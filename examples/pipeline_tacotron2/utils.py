from collections import defaultdict
import logging
import os
import shutil
from typing import List, Tuple
import json

import torch
from torch import Tensor

from text import text_to_sequence


class MetricLogger(defaultdict):
    def __init__(self, name, print_freq=1, disable=False):
        super().__init__(lambda: 0.0)
        self.disable = disable
        self.print_freq = print_freq
        self._iter = 0
        self["name"] = name

    def __str__(self):
        return json.dumps(self)

    def __call__(self):
        self._iter = (self._iter + 1) % self.print_freq
        if not self.disable and not self._iter:
            print(self, flush=True)


def save_checkpoint(state, is_best, filename):
    r"""Save the model to a temporary file first,
    then copy it to filename, in case the signal interrupts
    the torch.save() process.
    """

    if filename == "":
        return

    tempfile = filename + ".temp"

    # Remove tempfile in case interuption during the copying from tempfile to filename
    if os.path.isfile(tempfile):
        os.remove(tempfile)

    torch.save(state, tempfile)
    if os.path.isfile(tempfile):
        os.rename(tempfile, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth")
    logging.info("Checkpoint: saved")


def pad_sequences(batch: Tensor) -> Tuple[Tensor, Tensor]:
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x) for x in batch]),
        dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]]
        text_padded[i, :text.size(0)] = text

    return text_padded, input_lengths


def prepare_input_sequence(texts: List[str]) -> Tuple[Tensor, Tensor]:

    d = []
    for text in texts:
        d.append(torch.IntTensor(
            text_to_sequence(text, ['english_cleaners'])[:]))

    text_padded, input_lengths = pad_sequences(d)
    return text_padded, input_lengths
