import torch
import requests
import pytest


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank: int = 0):
        super().__init__()
        self.blank = blank
        self.labels = labels

    def forward(self, logits: torch.Tensor) -> str:
        """Given a sequence logits over labels, get the best path string

        Args:
            logits (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
            str: The resulting transcript
        """
        best_path = torch.argmax(logits, dim=-1)  # [num_seq,]
        best_path = torch.unique_consecutive(best_path, dim=-1)
        hypothesis = []
        for i in best_path:
            if i != self.blank:
                hypothesis.append(self.labels[i])
        return ''.join(hypothesis)


@pytest.fixture
def ctc_decoder():
    return GreedyCTCDecoder


_FILES = {
    'en': 'Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.flac',
    'de': '20090505-0900-PLENARY-16-de_20090505-21_56_00_8.flac',
    'es': '20130207-0900-PLENARY-7-es_20130207-13_02_05_5.flac',
    'fr': '20121212-0900-PLENARY-5-fr_20121212-11_37_04_10.flac',
    'it': '20170516-0900-PLENARY-16-it_20170516-18_56_31_1.flac',
}


@pytest.fixture
def sample_speech(tmp_path, lang):
    if lang not in _FILES:
        raise NotImplementedError(f'Unexpected lang: {lang}')
    filename = _FILES[lang]
    path = tmp_path.parent / filename
    if not path.exists():
        url = f'https://download.pytorch.org/torchaudio/test-assets/{filename}'
        print(f'downloading from {url}')
        with open(path, 'wb') as file:
            with requests.get(url) as resp:
                resp.raise_for_status()
                file.write(resp.content)
    return path
