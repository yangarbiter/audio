import random
from typing import List

import torch
from torch import Tensor
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torchaudio.models import get_pretrained_wavernn
from torchaudio.datasets import LJSPEECH
import numpy as np
from tqdm import tqdm

from processing import NormalizeDB


def fold_with_overlap(x, target, overlap):
    r'''Fold the tensor with overlap for quick batched inference.
    Overlap will be used for crossfading in xfade_and_unfold()

    Args:
        x (tensor): Upsampled conditioning features.
                        shape=(1, timesteps, features)
        target (int): Target timesteps for each index of batch
        overlap (int): Timesteps for both xfade and rnn warmup
    Return:
        (tensor) : shape=(num_folds, target + 2 * overlap, features)
    Details:
        x = [[h1, h2, ... hn]]
        Where each h is a vector of conditioning features
        Eg: target=2, overlap=1 with x.size(1)=10
        folded = [[h1, h2, h3, h4],
                  [h4, h5, h6, h7],
                  [h7, h8, h9, h10]]
    '''

    _, total_len, features = x.size()

    # Calculate variables needed
    num_folds = (total_len - overlap) // (target + overlap)
    extended_len = num_folds * (overlap + target) + overlap
    remaining = total_len - extended_len

    # Pad if some time steps poking out
    if remaining != 0:
        num_folds += 1
        padding = target + 2 * overlap - remaining
        x = pad_tensor(x, padding, side='after')

    folded = torch.zeros(num_folds, target + 2 * overlap, features, device=x.device)

    # Get the values for the folded tensor
    for i in range(num_folds):
        start = i * (target + overlap)
        end = start + target + 2 * overlap
        folded[i] = x[:, start:end, :]

    return folded

def xfade_and_unfold(y: Tensor, target: int, overlap: int) -> Tensor:
    ''' Applies a crossfade and unfolds into a 1d array.

    Args:
        y (Tensor): Batched sequences of audio samples
                    shape=(num_folds, target + 2 * overlap)
        target (int):
        overlap (int): Timesteps for both xfade and rnn warmup

    Returns:
        (Tensor) : audio samples in a 1d array
                    shape=(total_len)
                    dtype=np.float64
    Details:
        y = [[seq1],
                [seq2],
                [seq3]]
        Apply a gain envelope at both ends of the sequences
        y = [[seq1_in, seq1_target, seq1_out],
                [seq2_in, seq2_target, seq2_out],
                [seq3_in, seq3_target, seq3_out]]
        Stagger and add up the groups of samples:
        [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]
    '''

    num_folds, length = y.shape
    target = length - 2 * overlap
    total_len = num_folds * (target + overlap) + overlap

    # Need some silence for the rnn warmup
    silence_len = overlap // 2
    fade_len = overlap - silence_len
    silence = torch.zeros((silence_len), dtype=y.dtype, device=y.device)
    linear = torch.ones((silence_len), dtype=y.dtype, device=y.device)

    # Equal power crossfade
    t = torch.linspace(-1, 1, fade_len, dtype=y.dtype, device=y.device)
    fade_in = np.sqrt(0.5 * (1 + t))
    fade_out = np.sqrt(0.5 * (1 - t))

    # Concat the silence to the fades
    fade_in = torch.cat([silence, fade_in])
    fade_out = torch.cat([linear, fade_out])

    # Apply the gain to the overlap samples
    y[:, :overlap] *= fade_in
    y[:, -overlap:] *= fade_out

    unfolded = torch.zeros((total_len), dtype=y.dtype, device=y.device)

    # Loop to add up all the samples
    for i in range(num_folds):
        start = i * (target + overlap)
        end = start + target + 2 * overlap
        unfolded[start:end] += y[i]

    return unfolded

def get_gru_cell(gru):
    gru_cell = torch.nn.GRUCell(gru.input_size, gru.hidden_size)
    gru_cell.weight_hh.data = gru.weight_hh_l0.data
    gru_cell.weight_ih.data = gru.weight_ih_l0.data
    gru_cell.bias_hh.data = gru.bias_hh_l0.data
    gru_cell.bias_ih.data = gru.bias_ih_l0.data
    return gru_cell

def pad_tensor(x, pad, side='both'):
    # NB - this is just a quick method i need right now
    # i.e., it won't generalise to other shapes/dims
    b, t, c = x.size()
    total = t + 2 * pad if side == 'both' else t + pad
    padded = torch.zeros(b, total, c, device=x.device)
    if side == 'before' or side == 'both':
        padded[:, pad:pad + t, :] = x
    elif side == 'after':
        padded[:, :t, :] = x
    return padded

def infer(model, mel_specgram: Tensor, mulaw_decode: bool = True,
          batched: bool = True, target: int = 11000, overlap: int = 550) -> Tensor:
    r"""Inference

    Args:
        model (WaveRNN): The WaveRNN model.
        mel_specgram (Tensor): mel spectrogram with shape (n_mels, n_time)
        mulaw_decode (bool): 
        batched (bool): batch prediction
        target (int): (Default: ``11000``)
        overlap (int): (Default: ``550``)

    Returns:
        waveform (Tensor): Reconstructed wave form with shape (n_time, ).

    """
    device = mel_specgram.device
    dtype = mel_specgram.dtype

    output: List[Tensor] = []
    rnn1 = get_gru_cell(model.rnn1)
    rnn2 = get_gru_cell(model.rnn2)

    mel_specgram = mel_specgram.unsqueeze(0)
    mel_specgram = pad_tensor(mel_specgram.transpose(1, 2), pad=model.pad, side='both')
    mel_specgram, aux = model.upsample(mel_specgram.transpose(1, 2))

    mel_specgram, aux = mel_specgram.transpose(1, 2), aux.transpose(1, 2)

    if batched:
        mel_specgram = fold_with_overlap(mel_specgram, target, overlap)
        aux = fold_with_overlap(aux, target, overlap)

    b_size, seq_len, _ = mel_specgram.size()

    h1 = torch.zeros((b_size, model.n_rnn), device=device, dtype=dtype)
    h2 = torch.zeros((b_size, model.n_rnn), device=device, dtype=dtype)
    x = torch.zeros((b_size, 1), device=device, dtype=dtype)

    d = model.n_aux
    aux_split = [aux[:, :, d * i:d * (i + 1)] for i in range(4)]

    for i in tqdm(range(seq_len)):

        m_t = mel_specgram[:, i, :]

        a1_t, a2_t, a3_t, a4_t = \
            (a[:, i, :] for a in aux_split)

        x = torch.cat([x, m_t, a1_t], dim=1)
        x = model.fc(x)
        h1 = rnn1(x, h1)

        x = x + h1
        inp = torch.cat([x, a2_t], dim=1)
        h2 = rnn2(inp, h2)

        x = x + h2
        x = torch.cat([x, a3_t], dim=1)
        x = F.relu(model.fc1(x))

        x = torch.cat([x, a4_t], dim=1)
        x = F.relu(model.fc2(x))

        logits = model.fc3(x)

        posterior = F.softmax(logits, dim=1)
        distrib = torch.distributions.Categorical(posterior)

        sample = 2 * distrib.sample().float() / (model.n_classes - 1.) - 1.
        #sample = distrib.sample().float()
        output.append(sample)
        x = sample.unsqueeze(-1)

    output = torch.stack(output).transpose(0, 1).cpu()

    if batched:
        output = xfade_and_unfold(output, target, overlap)
    else:
        output = output[0]

    return output


def decode_mu_law(y: Tensor, mu: int, from_labels: bool = True) -> Tensor:
    if from_labels:
        y = 2 * y / (mu - 1.) - 1.
    mu = mu - 1
    x = torch.sign(y) / mu * ((1 + mu) ** torch.abs(y) - 1)
    return x


def main():
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dset = LJSPEECH("./", download=True)
    waveform, sample_rate, _, _ = dset[0]
    torchaudio.save("original.wav", waveform, sample_rate=sample_rate)

    n_bits = 8
    mel_kwargs = {
        'sample_rate': sample_rate,
        'n_fft': 2048,
        'f_min': 40.,
        'n_mels': 80,
        'win_length': 1100,
        'hop_length': 275,
        'mel_scale': 'slaney',
        'norm': 'slaney',
        'power': 1,
    }
    transforms = torch.nn.Sequential(
        MelSpectrogram(**mel_kwargs),
        NormalizeDB(min_level_db=-100, normalization=True),
    )
    mel_specgram = transforms(waveform)

    wavernn_model = get_pretrained_wavernn("wavernn_10k_epochs_8bits_ljspeech",
            progress=True).eval().to(device)
    wavernn_model.pad = (wavernn_model.kernel_size - 1) // 2

    with torch.no_grad():
        output = infer(wavernn_model, mel_specgram.to(device))

    output = torchaudio.functional.mu_law_decoding(output, n_bits)
    #output = decode_mu_law(output, 2**n_bits, False)

    torch.save(output, "output.pkl")

    torchaudio.save("result.wav", output.reshape(1, -1), sample_rate=sample_rate)
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
