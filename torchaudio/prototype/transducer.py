import torch
from torch.autograd import Function
from torch.nn import Module
from torchaudio._internal import (
    module_utils as _mod_utils,
)

__all__ = ["rnnt_loss", "RNNTLoss"]


class _RNNT(Function):
    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens, blank, reduction):
        """
        acts: Tensor of (batch x seqLength x labelLength x outputDim) containing output from network
        labels: 2 dimensional Tensor containing all the targets of the batch with zero padded
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        """

        device = acts.device
        certify_inputs(acts, labels, act_lens, label_lens)


        acts = acts.to("cpu")
        labels = labels.to("cpu")
        act_lens = act_lens.to("cpu")
        label_lens = label_lens.to("cpu")

        loss_func = torch.ops.warprnnt_pytorch_warp_rnnt.rnnt

        grads = torch.zeros_like(acts)
        minibatch_size = acts.size(0)
        costs = torch.zeros(minibatch_size, dtype=acts.dtype)

        loss_func(acts, labels, act_lens, label_lens, costs, grads, blank, 0)

        if reduction in ["sum", "mean"]:
            costs = costs.sum().unsqueeze_(-1)
            if reduction == "mean":
                costs /= minibatch_size
                grads /= minibatch_size

        costs = costs.to(device)
        ctx.grads = grads.to(device)

        return costs

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.view(-1, 1, 1, 1).to(ctx.grads)
        return ctx.grads.mul_(grad_output), None, None, None, None, None


@_mod_utils.requires_module('_warp_transducer')
def rnnt_loss(acts, labels, act_lens, label_lens, blank=0, reduction="mean"):
    """RNN Transducer Loss

    Args:
        acts: Tensor of (batch x seqLength x labelLength x outputDim) containing output from network
        labels: 2 dimensional Tensor containing all the targets of the batch with zero padded
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        blank (int, optional): blank label. Default: 0.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the output losses will be divided by the target lengths and
            then the mean over the batch is taken. Default: 'mean'
    """

    # NOTE manually done log_softmax for CPU version,
    # log_softmax is computed within GPU version.
    acts = torch.nn.functional.log_softmax(acts, -1)
    return _RNNT.apply(acts, labels, act_lens, label_lens, blank, reduction)


@_mod_utils.requires_module('_warp_transducer')
class RNNTLoss(Module):
    """
    Parameters:
        blank (int, optional): blank label. Default: 0.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the output losses will be divided by the target lengths and
            then the mean over the batch is taken. Default: 'mean'
    """

    def __init__(self, blank=0, reduction="mean"):
        super(RNNTLoss, self).__init__()
        self.blank = blank
        self.reduction = reduction
        self.loss = _RNNT.apply

    def forward(self, acts, labels, act_lens, label_lens):
        """
        acts: Tensor of (batch x seqLength x labelLength x outputDim) containing output from network
        labels: 2 dimensional Tensor containing all the targets of the batch with zero padded
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        """

        # NOTE manually done log_softmax for CPU version,
        # log_softmax is computed within GPU version.
        acts = torch.nn.functional.log_softmax(acts, -1)
        return self.loss(acts, labels, act_lens, label_lens, self.blank, self.reduction)


def check_type(var, t, name):
    if var.dtype is not t:
        raise TypeError("{} must be {}".format(name, t))


def check_contiguous(var, name):
    if not var.is_contiguous():
        raise ValueError("{} must be contiguous".format(name))


def check_dim(var, dim, name):
    if len(var.shape) != dim:
        raise ValueError("{} must be {}D".format(name, dim))


def certify_inputs(log_probs, labels, lengths, label_lengths):
    # check_type(log_probs, torch.float32, "log_probs")
    check_type(labels, torch.int32, "labels")
    check_type(label_lengths, torch.int32, "label_lengths")
    check_type(lengths, torch.int32, "lengths")
    check_contiguous(log_probs, "log_probs")
    check_contiguous(labels, "labels")
    check_contiguous(label_lengths, "label_lengths")
    check_contiguous(lengths, "lengths")

    if lengths.shape[0] != log_probs.shape[0]:
        raise ValueError("must have a length per example.")
    if label_lengths.shape[0] != log_probs.shape[0]:
        raise ValueError("must have a label length per example.")

    check_dim(log_probs, 4, "log_probs")
    check_dim(labels, 2, "labels")
    check_dim(lengths, 1, "lenghts")
    check_dim(label_lengths, 1, "label_lenghts")
    max_T = torch.max(lengths)
    max_U = torch.max(label_lengths)
    T, U = log_probs.shape[1:3]
    if T != max_T:
        raise ValueError("Input length mismatch")
    if U != max_U + 1:
        raise ValueError("Output length mismatch")