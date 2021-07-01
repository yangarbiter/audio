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
from math import sqrt
from typing import Tuple
import warnings

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        return self.conv(signal)


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).byte()
    mask = torch.le(mask, 0)
    return mask


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    """Attention model

    Args:
    """

    def __init__(self, attention_rnn_dim, embedding_dim,
                 attention_dim, attention_location_n_filters,
                 attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        Args:
            query: decoder output (batch, n_mels * n_frames_per_step)
            processed_memory: processed encoder outputs (B, T_in, attention_dim)
            attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        Returns:
            alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(2)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask) -> Tuple[Tensor, Tensor]:
        """
        Args:
            attention_hidden_state: attention rnn last output
            memory: encoder outputs
            processed_memory: processed encoder outputs
            attention_weights_cat: previous and cumulative attention weights
            mask: binary mask for padded data

        Returns:
            attanteion_context
            attention_weights
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        alignment = alignment.masked_fill(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, n_mels, postnet_embedding_dim,
                 postnet_kernel_size, postnet_n_convolutions):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(n_mels, postnet_embedding_dim,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(postnet_embedding_dim))
        )

        for _ in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(postnet_embedding_dim,
                             postnet_embedding_dim,
                             kernel_size=postnet_kernel_size, stride=1,
                             padding=int((postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(postnet_embedding_dim, n_mels,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(n_mels))
        )
        self.n_convs = len(self.convolutions)

    def forward(self, x):
        i = 0
        for conv in self.convolutions:
            if i < self.n_convs - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, training=self.training)
            else:
                x = F.dropout(conv(x), 0.5, training=self.training)
            i += 1

        return x


class Encoder(nn.Module):
    """
    Input: character embedding
    Output:

    Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, encoder_n_convolutions,
                 encoder_embedding_dim, encoder_kernel_size):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(encoder_embedding_dim,
                         encoder_embedding_dim,
                         kernel_size=encoder_kernel_size, stride=1,
                         padding=int((encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(encoder_embedding_dim,
                            int(encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x: Tensor, input_lengths: Tensor) -> Tensor:
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        input_lengths = input_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs


class Decoder(nn.Module):
    """
    attention
    """
    def __init__(self, n_mels, n_frames_per_step,
                 encoder_embedding_dim, attention_dim,
                 attention_location_n_filters,
                 attention_location_kernel_size,
                 attention_rnn_dim, decoder_rnn_dim,
                 prenet_dim, max_decoder_steps, gate_threshold,
                 p_attention_dropout, p_decoder_dropout,
                 early_stopping):
        super(Decoder, self).__init__()
        self.n_mels = n_mels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout
        self.early_stopping = early_stopping

        self.prenet = Prenet(
            n_mels * n_frames_per_step,
            [prenet_dim, prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            prenet_dim + encoder_embedding_dim,
            attention_rnn_dim)

        self.attention_layer = Attention(
            attention_rnn_dim, encoder_embedding_dim,
            attention_dim, attention_location_n_filters,
            attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            attention_rnn_dim + encoder_embedding_dim,
            decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim,
            n_mels * n_frames_per_step)

        self.gate_layer = LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as the first decoder input

        args:
            memory: decoder outputs
        
        returns:
            decoder_input: all zeros frames
        """

        B = memory.size(0)
        dtype = memory.dtype
        device = memory.device
        decoder_input = torch.zeros(
            B, self.n_mels * self.n_frames_per_step,
            dtype=dtype, device=device)
        return decoder_input

    def initialize_decoder_states(self,
                                  memory) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory

        Args:
            memory: Encoder outputs
        
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)
        dtype = memory.dtype
        device = memory.device

        attention_hidden = torch.zeros(
            B, self.attention_rnn_dim, dtype=dtype, device=device)
        attention_cell = torch.zeros(
            B, self.attention_rnn_dim, dtype=dtype, device=device)

        decoder_hidden = torch.zeros(
            B, self.decoder_rnn_dim, dtype=dtype, device=device)
        decoder_cell = torch.zeros(
            B, self.decoder_rnn_dim, dtype=dtype, device=device)

        attention_weights = torch.zeros(
            B, MAX_TIME, dtype=dtype, device=device)
        attention_weights_cum = torch.zeros(
            B, MAX_TIME, dtype=dtype, device=device)
        attention_context = torch.zeros(
            B, self.encoder_embedding_dim, dtype=dtype, device=device)

        processed_memory = self.attention_layer.memory_layer(memory)

        return (attention_hidden, attention_cell, decoder_hidden,
                decoder_cell, attention_weights, attention_weights_cum,
                attention_context, processed_memory)

    def _parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs
        RETURNS
        -------
        inputs: processed decoder inputs
        """
        # (B, n_mels, T_out) -> (B, T_out, n_mels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step), -1)
        # (B, T_out, n_mels) -> (T_out, B, n_mels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def _parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:
        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = alignments.transpose(0, 1).contiguous()
        # (T_out, B) -> (B, T_out)
        gate_outputs = gate_outputs.transpose(0, 1).contiguous()
        # (T_out, B, n_mels) -> (B, T_out, n_mels)
        mel_outputs = mel_outputs.transpose(0, 1).contiguous()
        # decouple frames per step
        shape = (mel_outputs.shape[0], -1, self.n_mels)
        mel_outputs = mel_outputs.view(*shape)
        # (B, T_out, n_mels) -> (B, n_mels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input, attention_hidden, attention_cell,
               decoder_hidden, decoder_cell, attention_weights,
               attention_weights_cum, attention_context, memory,
               processed_memory, mask):
        """ Decoder step using stored states, attention and memory

        Args:
            decoder_input: previous mel output

        Returns:
            mel_output:
            gate_output: gate output energies
            attention_weights:
        """
        cell_input = torch.cat((decoder_input, attention_context), -1)

        attention_hidden, attention_cell = self.attention_rnn(
            cell_input, (attention_hidden, attention_cell))
        attention_hidden = F.dropout(
            attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (attention_weights.unsqueeze(1),
             attention_weights_cum.unsqueeze(1)), dim=1)
        attention_context, attention_weights = self.attention_layer(
            attention_hidden, memory, processed_memory,
            attention_weights_cat, mask)

        attention_weights_cum += attention_weights
        decoder_input = torch.cat(
            (attention_hidden, attention_context), -1)

        decoder_hidden, decoder_cell = self.decoder_rnn(
            decoder_input, (decoder_hidden, decoder_cell))
        decoder_hidden = F.dropout(
            decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (decoder_hidden, attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return (decoder_output, gate_prediction, attention_hidden,
                attention_cell, decoder_hidden, decoder_cell, attention_weights,
                attention_weights_cum, attention_context)

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training

        Args:
            memory: Encoder outputs
            decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
            memory_lengths: Encoder output lengths for attention masking.
        
        Returns:
            mel_outputs: mel outputs from the decoder
            gate_outputs: gate outputs from the decoder
            alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self._parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        mask = get_mask_from_lengths(memory_lengths)
        (attention_hidden,
         attention_cell,
         decoder_hidden,
         decoder_cell,
         attention_weights,
         attention_weights_cum,
         attention_context,
         processed_memory) = self.initialize_decoder_states(memory)

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            (mel_output,
             gate_output,
             attention_hidden,
             attention_cell,
             decoder_hidden,
             decoder_cell,
             attention_weights,
             attention_weights_cum,
             attention_context) = self.decode(decoder_input,
                                              attention_hidden,
                                              attention_cell,
                                              decoder_hidden,
                                              decoder_cell,
                                              attention_weights,
                                              attention_weights_cum,
                                              attention_context,
                                              memory,
                                              processed_memory,
                                              mask)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self._parse_decoder_outputs(
            torch.stack(mel_outputs),
            torch.stack(gate_outputs),
            torch.stack(alignments))

        return mel_outputs, gate_outputs, alignments

    @torch.jit.export
    def infer(self, memory, memory_lengths):
        """ Decoder inference

        Args:
            memory: Encoder outputs

        Returns:
            mel_outputs: mel outputs from the decoder
            gate_outputs: gate outputs from the decoder
            alignments: sequence of attention weights from the decoder
            mel_lengths: 
        """
        decoder_input = self.get_go_frame(memory)

        mask = get_mask_from_lengths(memory_lengths)
        (attention_hidden,
         attention_cell,
         decoder_hidden,
         decoder_cell,
         attention_weights,
         attention_weights_cum,
         attention_context,
         processed_memory) = self.initialize_decoder_states(memory)

        mel_lengths = torch.zeros([memory.size(0)], dtype=torch.int32, device=memory.device)
        not_finished = torch.ones([memory.size(0)], dtype=torch.int32, device=memory.device)

        mel_outputs, gate_outputs, alignments = (
            torch.zeros(1), torch.zeros(1), torch.zeros(1))
        first_iter = True
        while True:
            decoder_input = self.prenet(decoder_input)
            (mel_output,
             gate_output,
             attention_hidden,
             attention_cell,
             decoder_hidden,
             decoder_cell,
             attention_weights,
             attention_weights_cum,
             attention_context) = self.decode(decoder_input,
                                              attention_hidden,
                                              attention_cell,
                                              decoder_hidden,
                                              decoder_cell,
                                              attention_weights,
                                              attention_weights_cum,
                                              attention_context,
                                              memory,
                                              processed_memory,
                                              mask)

            if first_iter:
                mel_outputs = mel_output.unsqueeze(0)
                gate_outputs = gate_output
                alignments = attention_weights
                first_iter = False
            else:
                mel_outputs = torch.cat(
                    (mel_outputs, mel_output.unsqueeze(0)), dim=0)
                gate_outputs = torch.cat((gate_outputs, gate_output), dim=0)
                alignments = torch.cat((alignments, attention_weights), dim=0)

            dec = torch.le(torch.sigmoid(gate_output),
                           self.gate_threshold).to(torch.int32).squeeze(1)

            not_finished = not_finished * dec
            mel_lengths += not_finished

            if self.early_stopping and torch.sum(not_finished) == 0:
                break
            if len(mel_outputs) == self.max_decoder_steps:
                warnings.warn("Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self._parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments, mel_lengths


class Tacotron2(nn.Module):
    r"""
    Tacotron2 model based on the implementation from `Nvidia <https://github.com/NVIDIA/DeepLearningExamples/>`_.

    The original implementation was introduced in
    *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions*
    [:footcite:`kalchbrenner2018efficient`]. The input channels of waveform and spectrogram have to be 1.
    The product of `upsample_scales` must equal `hop_length`.

    Args:
        mask_padding: (Default: ``False``)
        n_mels: number of mel bins (Default: ``80``)
        n_symbols: number of symbols for the input text (Default: ``148``)

        n_classes: the number of output classes.
        hop_length: the number of samples between the starts of consecutive frames.
        n_res_block: the number of ResBlock in stack. (Default: ``10``)
        n_rnn: the dimension of RNN layer. (Default: ``512``)
        n_fc: the dimension of fully connected layer. (Default: ``512``)
        kernel_size: the number of kernel size in the first Conv1d layer. (Default: ``5``)
        n_freq: the number of bins in a spectrogram. (Default: ``128``)
        n_hidden: the number of hidden dimensions of resblock. (Default: ``128``)
        n_output: the number of output dimensions of melresnet. (Default: ``128``)

    Example
        >>> tacotron2 = Tacotron2()
        >>> text = "Hello world!"
    """
    def __init__(self,
                 mask_padding: bool = False,
                 n_mels: int = 80,
                 n_symbols: int = 148,
                 symbols_embedding_dim: int = 512,
                 encoder_kernel_size: int = 5,
                 encoder_n_convolutions: int = 3,
                 encoder_embedding_dim: int = 512,
                 attention_rnn_dim: int = 1024,
                 attention_dim: int = 128,
                 attention_location_n_filters: int = 32,
                 attention_location_kernel_size: int = 31,
                 n_frames_per_step: int = 1,
                 decoder_rnn_dim: int = 1024,
                 prenet_dim: int = 256,
                 max_decoder_steps: int = 2000,
                 gate_threshold: float = 0.5,
                 p_attention_dropout: float = 0.1,
                 p_decoder_dropout: float = 0.1,
                 postnet_embedding_dim: int = 512,
                 postnet_kernel_size: int = 5,
                 postnet_n_convolutions: int = 5,
                 decoder_no_early_stopping: bool = True) -> None:
        super(Tacotron2, self).__init__()

        self.mask_padding = mask_padding
        self.n_mels = n_mels
        self.n_frames_per_step = n_frames_per_step
        self.embedding = nn.Embedding(n_symbols, symbols_embedding_dim)
        std = sqrt(2.0 / (n_symbols + symbols_embedding_dim))
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(encoder_n_convolutions,
                               encoder_embedding_dim,
                               encoder_kernel_size)
        self.decoder = Decoder(n_mels, n_frames_per_step,
                               encoder_embedding_dim, attention_dim,
                               attention_location_n_filters,
                               attention_location_kernel_size,
                               attention_rnn_dim, decoder_rnn_dim,
                               prenet_dim, max_decoder_steps,
                               gate_threshold, p_attention_dropout,
                               p_decoder_dropout,
                               not decoder_no_early_stopping)
        self.postnet = Postnet(n_mels, postnet_embedding_dim,
                               postnet_kernel_size,
                               postnet_n_convolutions)

    def _parse_output(self,
                      outputs: Tuple[Tensor, Tensor, Tensor],
                      output_lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        if self.mask_padding and output_lengths is not None:
            mask = get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].masked_fill_(mask, 0.0)
            outputs[1].masked_fill_(mask, 0.0)
            outputs[2].masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self,
                text: Tensor,
                text_lengths: Tensor,
                mel_specgram: Tensor,
                mel_specgram_lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Pass the input through the Tacotron2 model. This is in teacher
        forcing mode, which is generally used for training.

        The input `text` should be padded with zeros to length text_lengths.max().
        The input `mel_specgram` should be padded with zeros to length mel_specgram_lengths.max().

        Args:
            text (Tensor): the input text to Tacotron2.  (n_batch, text_lengths.max())
            text_lengths (Tensor): the length of each text (n_batch)
            mel_specgram (Tensor): the target mel spectrogram (n_batch, n_mels, time=mel_specgram_lengths.max())
            mel_specgram_lengths (Tensor): the length of each mel spectrogram (n_batch)

        Return:
            mel_specgram (Tensor): mel spectrogram before postnet (n_batch, n_mels, mel_specgram_lengths.max())
            mel_specgram_postnet (Tensor): mel spectrogram after postnet (n_batch, n_mels, mel_specgram_lengths.max())
            stop_token (Tensor): the output for stop token at each time step (n_batch, mel_specgram_lengths.max())
        """

        text_lengths, mel_specgram_lengths = text_lengths.data, mel_specgram_lengths.data

        embedded_inputs = self.embedding(text).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mel_specgram, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self._parse_output(
            (mel_outputs, mel_outputs_postnet, gate_outputs, alignments),
            mel_specgram_lengths)

    def infer(self, text: Tensor, text_lengths: Tensor) -> Tensor:
        r"""Using Tacotron2 for inference. This is generally used for generating mel spectrograms.

        The input `text` should be padded with zeros to length text_lengths.max().

        Args:
            text (Tensor): the input text to Tacotron2.  (n_batch, text_lengths.max())
            text_lengths (Tensor): the length of each text (n_batch)

        Return:
            mel_specgram (Tensor): the predicted mel spectrogram (n_batch, n_mels, mel_specgram_lengths.max())
        """

        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        mel_outputs, _, _, _ = self.decoder.infer(encoder_outputs, text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        # BS = mel_outputs_postnet.size(0)
        # alignments = alignments.unfold(1, BS, BS).transpose(0,2)

        # return mel_outputs_postnet, mel_lengths, alignments
        return mel_outputs_postnet
