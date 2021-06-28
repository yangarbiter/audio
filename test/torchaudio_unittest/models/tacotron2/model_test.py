import torch
import torch.nn.functional as F

from parameterized import parameterized

from torchaudio.models.tacotron2 import (
    Tacotron2
)
from ...common_utils import (
    TorchaudioTestCase,
    skipIfNoCuda,
)


class TestTacotron2Model(TorchaudioTestCase):
    def _smoke_test(self, device, dtype):
        torch.manual_seed(0)

        n_mels = 80 
        max_decoder_steps = 100
        model = Tacotron2(
            mask_padding=False,
            n_mels=n_mels,
            n_symbols=148,
            symbols_embedding_dim=512,
            encoder_kernel_size=5,
            encoder_n_convolutions=3,
            encoder_embedding_dim=512,
            attention_rnn_dim=1024,
            attention_dim=128,
            attention_location_n_filters=32,
            attention_location_kernel_size=31,
            n_frames_per_step=1,
            decoder_rnn_dim=1024,
            prenet_dim=256,
            max_decoder_steps=max_decoder_steps,
            gate_threshold=0.5,
            p_attention_dropout=0.1,
            p_decoder_dropout=0.1,
            postnet_embedding_dim=512,
            postnet_kernel_size=5,
            postnet_n_convolutions=5,
            decoder_no_early_stopping=True,
        )
        model = model.to(device=device, dtype=dtype)
        model = model.eval()

        # "hellow world"
        text_padded = torch.as_tensor([[45, 42, 49, 49, 52, 60, 11, 60, 52, 55, 49, 41]], dtype=torch.int64).to(device)
        input_lengths = torch.as_tensor([12], dtype=torch.int64).to(device)

        with torch.no_grad():
            mel_specgram = model.infer(text_padded, input_lengths)

        assert mel_specgram.size() == (1, n_mels, max_decoder_steps)

        self.assertEqual(
            mel_specgram[0][0][:5],
            torch.as_tensor([-0.0183,  0.2122,  0.1738,  0.2315,  0.2268], dtype=dtype),
            rtol=1e-02,
            atol=1e-02,
        )


    @parameterized.expand([(torch.float32, ), (torch.float64, )])
    def test_cpu_smoke_test(self, dtype):
        self._smoke_test(torch.device('cpu'), dtype)

    @parameterized.expand([(torch.float32, ), (torch.float64, )])
    @skipIfNoCuda
    def test_cuda_smoke_test(self, dtype):
        self._smoke_test(torch.device('cuda'), dtype)