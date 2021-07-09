# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from typing import Tuple

from torch import nn, Tensor


class Tacotron2Loss(nn.Module):
    """Tacotron2 loss function adapted from:
    https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/tacotron2/loss_function.py
    """

    def __init__(self):
        super().__init__()

    def forward(self,
                model_outputs: Tuple[Tensor, Tensor, Tensor],
                targets: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Pass the input through the Tacotron2 loss.

        Args:
            model_outputs (tuple of three Tensors): The outputs of the
                Tacotron2. These outputs should include three items:
                (1) the predicted mel spectrogram before the postnet (mel_specgram) with shape (n_batch, n_mel, n_time),
                (2) predicted mel spectrogram after the postnet (mel_specgram_postnet)
                    with shape (n_batch, n_mel, n_time), and
                (3) the stop token prediction (gate_out) with shape (n_batch).
            targets (tuple of two Tensors): The ground truth mel spectrogram (n_batch, n_mel, n_time) and
                stop token with shape (n_batch).

        Returns:
            mel_loss (Tensor): The mean MSE of the mel_specgram and ground truth mel spectrogram with shape (n_batch, ).
            mel_postnet_loss (Tensor): The mean MSE of the mel_specgram_postnet and
                ground truth mel spectrogram with shape (n_batch, ).
            gate_loss (Tensor): The mean binary cross entropy loss of
                the prediction on the stop token with shape (n_batch, ).
        """
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_specgram, mel_specgram_postnet, gate_out = model_outputs
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss(reduction="mean")(mel_specgram, mel_target)
        mel_postnet_loss = nn.MSELoss(reduction="mean")(mel_specgram_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss(reduction="mean")(gate_out, gate_target)
        return mel_loss, mel_postnet_loss, gate_loss
