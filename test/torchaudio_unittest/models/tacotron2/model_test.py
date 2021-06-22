import torch
import torch.nn.functional as F

from parameterized import parameterized

from torchaudio.models.tacotron2 import (
    Tacotron2
)
from torchaudio.models.tacotron2.utils import Processing
from ...common_utils import (
    TorchaudioTestCase,
    skipIfNoQengine,
    skipIfNoCuda,
)


class TestTacotron2Model(TorchaudioTestCase):
    def _smoke_test(self, device, dtype):
        model = Tacotron2()
        model = model.to(device=device, dtype=dtype)
        model = model.eval()

        torch.manual_seed(0)

        text = "Hello world."
        cpu_run = (device == torch.device('cpu'))
        text_padded, input_lengths = Processing.prepare_input_sequence(text, cpu_run)
        model.infer(text_padded, input_lengths)

    @parameterized.expand([(torch.float32, ), (torch.float64, )])
    def test_cpu_smoke_test(self, dtype):
        self._smoke_test(torch.device('cpu'), dtype)

    @parameterized.expand([(torch.float32, ), (torch.float64, )])
    @skipIfNoCuda
    def test_cuda_smoke_test(self, dtype):
        self._smoke_test(torch.device('cuda'), dtype)