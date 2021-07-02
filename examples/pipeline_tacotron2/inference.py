import random
import argparse

import torch
import torchaudio
import numpy as np
from torchaudio.models.tacotron2 import Tacotron2

from utils import prepare_input_sequence
from datasets import InverseSpectralNormalization


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument(
        '--input-text',
        '-i',
        type=str,
        default="Hello world.",
        help='[string] Type in something here and TTS will generate it!'
    )
    parser.add_argument(
        '--vocoder',
        default='nvidia_waveglow',
        choices=['griffin_lim', 'nvidia_waveglow'],
        type=str,
        help="select the vocoder to use",
    )
    return parser

def unwrap_distributed(state_dict):
    """Unwraps model from DistributedDataParallel. DDP wraps model in
    additional "module.", it needs to be removed for single GPU inference.

    Adapted from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/inference.py

    Args:
        state_dict: model's state dict

    Return:
        unwrapped_state_dict: model's state dict for single GPU
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.1.', '')
        new_key = new_key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict

def main(args):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sample_rate = 22050

    tacotron2 = Tacotron2()
    tacotron2.load_state_dict(unwrap_distributed(torch.load("./test_ckpt.pth", map_location="cuda")['state_dict']))
    tacotron2 = tacotron2.to(device)
    tacotron2.eval()

    #sequences, lengths = prepare_input_sequence([args.input_text])
    import ipdb; ipdb.set_trace()
    sequences, lengths = prepare_input_sequence([
        "Hello world I missed you so much ",
        "Ken is so funny",
    ])
    sequences, lengths = sequences.long().to(device), lengths.long().to(device)
    with torch.no_grad():
        mel_specgram, _ = tacotron2.infer(sequences, lengths)

    if args.vocoder == "nvidia_waveglow":
        waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow = waveglow.to(device)
        waveglow.eval()

        with torch.no_grad():
            waveform = waveglow.infer(mel_specgram).cpu()

    elif args.vocoder == "griffin_lim":
        from torchaudio.transforms import GriffinLim, InverseMelScale

        inv_norm = InverseSpectralNormalization()
        inv_mel = InverseMelScale(
            n_stft=(1024 // 2 + 1),
            n_mels=80,
            sample_rate=sample_rate,
            f_min=0.,
            f_max=8000.,
            mel_scale="slaney",
            norm='slaney',
        )
        griffin_lim = GriffinLim(
            n_fft=1024,
            power=1,
            hop_length=256,
            win_length=1024,
        )

        specgram = inv_mel(inv_norm(mel_specgram.cpu()))
        waveform = griffin_lim(specgram)

    torchaudio.save("audio.wav", waveform, sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    main(args)
