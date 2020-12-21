#!/usr/bin/env python3

import time
import unittest

import numpy as np
import torch
from torchaudio.transducer import RNNTLoss as PyTorchTransducerLoss
from torchaudio.transducer import RNNTLoss as WarpTransducerLoss
from torchaudio_unittest import common_utils

from .numpy_transducer import (
    AlignmentRestrictionCheck,
    NumpyTransducerLoss,
    _NumpyTransducer,
)


DEFAULT_CUDA_DEVICE = torch.device("cuda")


def get_sparse(data, dense_tensor, l_buffer, r_buffer, H=1):
    B, _, U, D = 2, 10, 3, 4
    total_valid = 0
    valid_ranges = np.zeros((B * H, U, 2), dtype=np.int32)
    cells_per_sample = np.zeros(B * H, dtype=np.int32)
    wp_ends = data["wp_ends"]
    for b_tgt in range(B * H):
        b_src = int(b_tgt / H)
        src_len = int(data["src_lengths"][b_src])
        tgt_len = int(data["tgt_lengths"][b_tgt]) + 1
        ar_check = AlignmentRestrictionCheck(
            tgt_len, src_len, wp_ends[b_tgt][:tgt_len], l_buffer, r_buffer
        )
        sample_cells = 0
        for u in range(tgt_len):
            v_range = ar_check.valid_time_ranges(u)
            valid_ranges[b_tgt, u, 0] = v_range[0]
            valid_ranges[b_tgt, u, 1] = v_range[1]
            total_valid += v_range[1] - v_range[0] + 1
            sample_cells += v_range[1] - v_range[0] + 1
        cells_per_sample[b_tgt] = sample_cells
    sparse_joint_enc = np.zeros((total_valid, D)).astype(dense_tensor.dtype)
    offset = 0
    for b in range(B * H):
        for u in range(U):
            st, en = valid_ranges[b_tgt][u][0], valid_ranges[b_tgt][u][1]
            sparse_joint_enc[offset : offset + (en - st) + 1, :] = dense_tensor[
                b, st : en + 1, u, :
            ]
            offset += (en - st) + 1
    return sparse_joint_enc, valid_ranges, cells_per_sample


def compute_with_pytorch_transducer(data):

    costs = PyTorchTransducerLoss(blank=data["blank"], reduction="none")(
        acts=data["logits_sparse"] if "logits_sparse" in data else data["logits"],
        labels=data["targets"],
        act_lens=data["src_lengths"],
        label_lens=data["tgt_lengths"],
    )

    loss = torch.sum(costs)
    loss.backward()
    costs = costs.cpu().data.numpy()
    gradients = data["logits"].saved_grad.cpu().data.numpy()
    return costs, gradients


def assert_sparse_all_close(data, gradients, ref_gradients, atol=1e-6, rtol=1e-2):
    valid_ranges = data["valid_ranges"]
    idx = 0
    for b in range(valid_ranges.size(0)):
        for u in range(valid_ranges.size(1)):
            st, en = valid_ranges[b, u, 0], valid_ranges[b, u, 1]
            np.testing.assert_allclose(
                gradients[idx : idx + (en - st + 1), :],
                ref_gradients[b, st : en + 1, u, :],
                atol=atol,
                rtol=rtol,
            )
            idx += (en - st) + 1


class PyTorchTransducerLossTest(unittest.TestCase):
    @staticmethod
    def _get_numpy_data_B2_T4_U3_D3(dtype=np.float32):
        logits = np.array(
            [
                0.065357,
                0.787530,
                0.081592,  # noqa
                0.529716,
                0.750675,
                0.754135,  # noqa
                0.609764,
                0.868140,
                0.622532,  # noqa
                0.668522,
                0.858039,
                0.164539,  # noqa
                0.989780,
                0.944298,
                0.603168,  # noqa
                0.946783,
                0.666203,
                0.286882,  # noqa
                0.094184,
                0.366674,
                0.736168,  # noqa
                0.166680,
                0.714154,
                0.399400,  # noqa
                0.535982,
                0.291821,
                0.612642,  # noqa
                0.324241,
                0.800764,
                0.524106,  # noqa
                0.779195,
                0.183314,
                0.113745,  # noqa
                0.240222,
                0.339470,
                0.134160,  # noqa
                0.505562,
                0.051597,
                0.640290,  # noqa
                0.430733,
                0.829473,
                0.177467,  # noqa
                0.320700,
                0.042883,
                0.302803,  # noqa
                0.675178,
                0.569537,
                0.558474,  # noqa
                0.083132,
                0.060165,
                0.107958,  # noqa
                0.748615,
                0.943918,
                0.486356,  # noqa
                0.418199,
                0.652408,
                0.024243,  # noqa
                0.134582,
                0.366342,
                0.295830,  # noqa
                0.923670,
                0.689929,
                0.741898,  # noqa
                0.250005,
                0.603430,
                0.987289,  # noqa
                0.592606,
                0.884672,
                0.543450,  # noqa
                0.660770,
                0.377128,
                0.358021,  # noqa
            ],
            dtype=dtype,
        ).reshape(2, 4, 3, 3)

        targets = np.array([[1, 2], [1, 1]], dtype=np.int32)
        src_lengths = np.array([4, 4], dtype=np.int32)
        tgt_lengths = np.array([2, 2], dtype=np.int32)

        blank = 0

        ref_costs = np.array([4.2806528590890736, 3.9384369822503591], dtype=dtype)

        ref_gradients = np.array(
            [
                -0.186844,
                -0.062555,
                0.249399,  # noqa
                -0.203377,
                0.202399,
                0.000977,  # noqa
                -0.141016,
                0.079123,
                0.061893,  # noqa
                -0.011552,
                -0.081280,
                0.092832,  # noqa
                -0.154257,
                0.229433,
                -0.075176,  # noqa
                -0.246593,
                0.146405,
                0.100188,  # noqa
                -0.012918,
                -0.061593,
                0.074512,  # noqa
                -0.055986,
                0.219831,
                -0.163845,  # noqa
                -0.497627,
                0.209240,
                0.288387,  # noqa
                0.013605,
                -0.030220,
                0.016615,  # noqa
                0.113925,
                0.062781,
                -0.176706,  # noqa
                -0.667078,
                0.367659,
                0.299419,  # noqa
                -0.356344,
                -0.055347,
                0.411691,  # noqa
                -0.096922,
                0.029459,
                0.067463,  # noqa
                -0.063518,
                0.027654,
                0.035863,  # noqa
                -0.154499,
                -0.073942,
                0.228441,  # noqa
                -0.166790,
                -0.000088,
                0.166878,  # noqa
                -0.172370,
                0.105565,
                0.066804,  # noqa
                0.023875,
                -0.118256,
                0.094381,  # noqa
                -0.104707,
                -0.108934,
                0.213642,  # noqa
                -0.369844,
                0.180118,
                0.189726,  # noqa
                0.025714,
                -0.079462,
                0.053748,  # noqa
                0.122328,
                -0.238789,
                0.116460,  # noqa
                -0.598687,
                0.302203,
                0.296484,  # noqa
            ],
            dtype=dtype,
        ).reshape(2, 4, 3, 3)

        data = {
            "logits": logits,
            "targets": targets,
            "src_lengths": src_lengths,
            "tgt_lengths": tgt_lengths,
            "blank": blank,
        }

        return data, ref_costs, ref_gradients

    @staticmethod
    def _get_numpy_data_B1_T2_U3_D5(dtype=np.float32):
        logits = np.array(
            [
                0.1,
                0.6,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.6,
                0.1,
                0.1,
                0.1,
                0.1,
                0.2,
                0.8,
                0.1,
                0.1,
                0.6,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.2,
                0.1,
                0.1,
                0.7,
                0.1,
                0.2,
                0.1,
                0.1,
            ],
            dtype=dtype,
        ).reshape(1, 2, 3, 5)
        targets = np.array([[1, 2]], dtype=np.int32)
        src_lengths = np.array([2], dtype=np.int32)
        tgt_lengths = np.array([2], dtype=np.int32)

        blank = -1

        ref_costs = np.array([5.09566688538], dtype=dtype)
        ref_gradients = np.array(
            [
                0.17703132,
                -0.39992708,
                0.17703132,
                0.17703132,
                -0.13116692,  # noqa
                0.12247062,
                0.12247062,
                -0.181684,
                0.12247062,
                -0.1857276,  # noqa
                0.06269141,
                0.06269141,
                0.06928471,
                0.12624498,
                -0.32091248,  # noqa
                0.05456069,
                -0.2182428,
                0.05456069,
                0.05456069,
                0.05456069,  # noqa
                0.12073967,
                0.12073967,
                -0.48295838,
                0.12073967,
                0.12073967,  # noqa
                0.30741188,
                0.16871123,
                0.18645471,
                0.16871123,
                -0.83128875,  # noqa
            ],
            dtype=dtype,
        ).reshape(1, 2, 3, 5)

        data = {
            "logits": logits,
            "targets": targets,
            "src_lengths": src_lengths,
            "tgt_lengths": tgt_lengths,
            "blank": blank,
        }

        return data, ref_costs, ref_gradients

    @staticmethod
    def _get_numpy_random_data(
        max_B=8, max_T=128, max_U=32, max_D=40, blank=-1, dtype=np.float32, seed=None
    ):
        if seed is not None:
            np.random.seed(seed=seed)

        if blank != -1:  # TODO(cfyeh): support blank != -1.
            raise ValueError("blank != -1 is not supported yet.")

        B = np.random.randint(low=1, high=max_B)
        T = np.random.randint(low=5, high=max_T)
        U = np.random.randint(low=5, high=max_U)
        D = np.random.randint(low=2, high=max_D)

        src_lengths = np.random.randint(low=5, high=T + 1, size=(B,), dtype=np.int32)
        tgt_lengths = np.random.randint(low=5, high=U + 1, size=(B,), dtype=np.int32)
        max_src_length = np.max(src_lengths)
        max_tgt_length = np.max(tgt_lengths)
        # TODO(cfyeh): support blank != -1.
        targets = np.random.randint(
            low=0, high=D - 1, size=(B, max_tgt_length), dtype=np.int32
        )
        logits = np.random.random_sample(
            size=(B, max_src_length, max_tgt_length + 1, D)
        ).astype(dtype=dtype)

        return {
            "logits": logits,
            "targets": targets,
            "src_lengths": src_lengths,
            "tgt_lengths": tgt_lengths,
            "blank": blank,
        }

    @staticmethod
    def _get_benchmarking_data(
        B=8,
        max_T=250,
        max_U=150,
        D=2048,
        blank=-1,
        dtype=torch.float32,
        device=DEFAULT_CUDA_DEVICE,
        seed=None,
    ):
        if seed is not None:
            torch.manual_seed(seed=seed)

        if blank != -1:  # TODO(cfyeh): support blank != -1.
            raise ValueError("blank != -1 is not supported yet.")

        # TODO(cfyeh): make device configurable.
        logits = torch.randn(
            B, max_T, max_U, D, dtype=dtype, device=device, requires_grad=True
        )
        src_lengths = torch.full(
            size=(B,), fill_value=max_T, dtype=torch.int32, device=device
        )
        tgt_lengths = torch.full(
            size=(B,), fill_value=(max_U - 1), dtype=torch.int32, device=device
        )
        # TODO(cfyeh): support blank != -1.
        targets = torch.randint(
            low=0, high=D - 1, size=(B, max_U - 1), dtype=torch.int32, device=device
        )

        data = {
            "logits": logits,
            "targets": targets,
            "src_lengths": src_lengths,
            "tgt_lengths": tgt_lengths,
            "blank": blank,
        }
        return data

    @staticmethod
    def _numpy_to_torch(data, device, requires_grad=True):
        logits = torch.from_numpy(data["logits"])
        targets = torch.from_numpy(data["targets"])
        src_lengths = torch.from_numpy(data["src_lengths"])
        tgt_lengths = torch.from_numpy(data["tgt_lengths"])

        if "wp_ends" in data:
            data["wp_ends"] = torch.from_numpy(data["wp_ends"]).to(device=device)

        logits = torch.autograd.Variable(logits, requires_grad=requires_grad)
        src_lengths = torch.autograd.Variable(src_lengths)
        tgt_lengths = torch.autograd.Variable(tgt_lengths)
        targets = torch.autograd.Variable(targets)

        if device == "cpu":
            logits = logits.cpu()
        elif device == "cuda":
            logits = logits.cuda()
        else:
            raise ValueError("unrecognized device = {}".format(device))

        def grad_hook(grad):
            logits.saved_grad = grad.clone()

        logits.register_hook(grad_hook)

        data["logits"] = logits
        data["src_lengths"] = src_lengths
        data["tgt_lengths"] = tgt_lengths
        data["targets"] = targets

        if "logits_sparse" in data:
            logits_sparse = torch.from_numpy(data["logits_sparse"])
            logits_sparse = torch.autograd.Variable(
                logits_sparse, requires_grad=requires_grad
            )
            logits_sparse = logits_sparse.to(device=logits.device)
            logits_sparse.register_hook(grad_hook)
            data["logits_sparse"] = logits_sparse
            valid_ranges = torch.from_numpy(data["valid_ranges"])
            valid_ranges = valid_ranges.to(device=logits.device)
            data["valid_ranges"] = valid_ranges
            cells_per_sample = torch.from_numpy(data["cells_per_sample"])
            cells_per_sample = cells_per_sample.to(device=logits.device)
            data["cells_per_sample"] = cells_per_sample
        return data

    @staticmethod
    def _compute_with_numpy_transducer(data):
        costs = NumpyTransducerLoss(
            blank=data["blank"],
        )(
            logits=data["logits"],
            src_lengths=data["src_lengths"],
            tgt_lengths=data["tgt_lengths"],
            targets=data["targets"],
            wp_ends=data["wp_ends"] if "wp_ends" in data else None,
            l_buffer=data["l_buffer"] if "l_buffer" in data else 0,
            r_buffer=data["r_buffer"] if "r_buffer" in data else 0,
        )

        loss = torch.sum(costs)
        loss.backward()

        costs = costs.cpu().data.numpy()
        gradients = data["logits"].saved_grad.cpu().data.numpy()

        return costs, gradients

    def _test_costs_and_gradients(
        self, data, ref_costs, ref_gradients, atol=1e-6, rtol=1e-2
    ):
        logits_shape = data["logits"].shape
        costs, gradients = compute_with_pytorch_transducer(data=data)
        np.testing.assert_allclose(costs, ref_costs, atol=atol, rtol=rtol)
        if "logits_sparse" in data:
            assert_sparse_all_close(
                data, gradients, ref_gradients, atol=atol, rtol=rtol
            )
        else:
            self.assertEqual(logits_shape, gradients.shape)
            if not np.allclose(gradients, ref_gradients, atol=atol, rtol=rtol):
                for b in range(len(gradients)):
                    T = data["src_lengths"][b]
                    U = data["tgt_lengths"][b]
                    for t in range(gradients.shape[1]):
                        for u in range(gradients.shape[2]):
                            np.testing.assert_allclose(
                                gradients[b, t, u],
                                ref_gradients[b, t, u],
                                atol=atol,
                                rtol=rtol,
                                err_msg=f"failed on b={b}, t={t}/T={T}, u={u}/U={U}",
                            )

    @common_utils.skipIfNoCuda
    def test_nan_logits(self):
        for sparse in [False, True]:
            data = self._get_B1_T10_U3_D4_data(
                random=True, l_buffer=10, r_buffer=10, sparse=sparse, nan=True
            )
            data = self._numpy_to_torch(
                data=data, device="cuda", requires_grad=True
            )
            data["l_buffer"] = 10
            data["r_buffer"] = 10
            costs, gradients = compute_with_pytorch_transducer(data=data)
            self.assertTrue(np.all(costs == 0))
            self.assertTrue(np.all(gradients == 0))

    # def test_costs_and_gradients_B1_T2_U3_D5_fp32_cpu(self):
    #     data, ref_costs, ref_gradients = self._get_numpy_data_B1_T2_U3_D5(
    #         dtype=np.float32
    #     )
    #     data = self._numpy_to_torch(data=data, device="cpu", requires_grad=True)
    #     self._test_costs_and_gradients(
    #         data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
    #     )

    # @common_utils.skipIfNoCuda
    # def test_costs_and_gradients_B1_T2_U3_D5_fp32_cuda(self):

    #     data, ref_costs, ref_gradients = self._get_numpy_data_B1_T2_U3_D5(
    #         dtype=np.float32
    #     )
    #     data = self._numpy_to_torch(data=data, device="cuda", requires_grad=True)
    #     self._test_costs_and_gradients(
    #         data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
    #     )

    def test_costs_and_gradients_B2_T4_U3_D3_fp32_cpu(self):
        data, ref_costs, ref_gradients = self._get_numpy_data_B2_T4_U3_D3(
            dtype=np.float32
        )
        data = self._numpy_to_torch(data=data, device="cpu", requires_grad=True)
        self._test_costs_and_gradients(
            data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
        )

    # @common_utils.skipIfNoCuda
    # def test_costs_and_gradients_B2_T4_U3_D3_fp32_cuda(self):

    #     data, ref_costs, ref_gradients = self._get_numpy_data_B2_T4_U3_D3(
    #         dtype=np.float32
    #     )
    #     data = self._numpy_to_torch(data=data, device="cuda", requires_grad=True)
    #     self._test_costs_and_gradients(
    #         data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
    #     )

    # def test_costs_and_gradients_random_data_with_numpy_fp32_cpu(self):
    #     seed = 777
    #     for i in range(5):
    #         data = self._get_numpy_random_data(dtype=np.float32, seed=(seed + i))
    #         data = self._numpy_to_torch(data=data, device="cpu", requires_grad=True)
    #         ref_costs, ref_gradients = self._compute_with_numpy_transducer(data=data)
    #         self._test_costs_and_gradients(
    #             data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
    #         )

    # @common_utils.skipIfNoCuda
    # def test_costs_and_gradients_random_data_with_numpy_fp32_cuda(self):

    #     seed = 777
    #     for i in range(5):
    #         data = self._get_numpy_random_data(dtype=np.float32, seed=(seed + i))
    #         data = self._numpy_to_torch(data=data, device="cuda", requires_grad=True)
    #         ref_costs, ref_gradients = self._compute_with_numpy_transducer(data=data)
    #         self._test_costs_and_gradients(
    #             data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
    #         )

    @staticmethod
    def _test_against_warp_transducer(data, iters):
        logits = data["logits"]
        src_lengths = data["src_lengths"]
        tgt_lengths = data["tgt_lengths"]
        targets = data["targets"]
        blank = data["blank"]

        warp_elapsed = 0
        for i in range(iters):
            tic = time.time()
            costs = WarpTransducerLoss(blank=blank)(
                logits, targets, src_lengths, tgt_lengths
            )
            loss = torch.sum(costs)
            loss.backward()
            toc = time.time()
            warp_elapsed = warp_elapsed + (toc - tic)
            print("WarpTransducer[{}] = {} secs".format(i, toc - tic))

        pytorch_elapsed = 0
        for i in range(iters):
            tic = time.time()
            costs = PyTorchTransducerLoss(blank=blank, reduction="none")(
                logits=logits,
                src_lengths=src_lengths,
                tgt_lengths=tgt_lengths,
                targets=targets,
            )
            loss = torch.sum(costs)
            loss.backward()
            toc = time.time()
            pytorch_elapsed = pytorch_elapsed + (toc - tic)
            print("PyTorchTransducer[{}] = {} secs".format(i, toc - tic))

        print(
            "\nWarpTransducer: {}/{} = {} secs".format(
                warp_elapsed, iters, (warp_elapsed / iters)
            )
        )
        print(
            "\nPyTorchTransducer: {}/{} = {} secs".format(
                pytorch_elapsed, iters, (pytorch_elapsed / iters)
            )
        )

    @staticmethod
    def _test_fp32_fp16(data, iters):
        src_lengths = data["src_lengths"]
        tgt_lengths = data["tgt_lengths"]
        targets = data["targets"]
        blank = data["blank"]

        fp32_logits = data["logits"].float()
        fp16_logits = data["logits"].half()

        fp32_elapsed = 0
        for i in range(iters):
            tic = time.time()
            costs = PyTorchTransducerLoss(blank=blank, reduction="none")(
                logits=fp32_logits,
                src_lengths=src_lengths,
                tgt_lengths=tgt_lengths,
                targets=targets,
            )
            loss = torch.sum(costs)
            loss.backward()
            toc = time.time()
            fp32_elapsed = fp32_elapsed + (toc - tic)
            print("fp32[{}] = {} secs".format(i, toc - tic))

        fp16_elapsed = 0
        for i in range(iters):
            tic = time.time()
            costs = PyTorchTransducerLoss(blank=blank, reduction="none")(
                logits=fp16_logits,
                src_lengths=src_lengths,
                tgt_lengths=tgt_lengths,
                targets=targets,
            )
            loss = torch.sum(costs)
            loss.backward()
            toc = time.time()
            fp16_elapsed = fp16_elapsed + (toc - tic)
            print("fp16[{}] = {} secs".format(i, toc - tic))

        print(
            "\nfp32: {}/{} = {} secs".format(
                fp32_elapsed, iters, (fp32_elapsed / iters)
            )
        )
        print(
            "\nfp16: {}/{} = {} secs".format(
                fp16_elapsed, iters, (fp16_elapsed / iters)
            )
        )

    def test_alignment_restricted_transducer_B2_T4_U3_D3_cpu(self):
        # Note - this test just ensures that the numpy and cpu c++
        # implementations match.
        # Probably a hand constructed test for gradients will be more thorough
        data, _, _ = self._get_numpy_data_B2_T4_U3_D3(dtype=np.float32)
        data = self._numpy_to_torch(data=data, device="cpu", requires_grad=True)
        data["wp_ends"] = torch.tensor([[0, 1, 2], [0, 1, 2]]).int()

        for l_buffer in [0]:
            for r_buffer in [0, 1, 2]:
                data["l_buffer"] = l_buffer
                data["r_buffer"] = r_buffer
                ref_costs, ref_gradients = self._compute_with_numpy_transducer(
                    data=data
                )

                self._test_costs_and_gradients(
                    data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
                )

    def test_alignment_restricted_transducer_B1_T10_U3_D4_cpu(self):
        data = self._get_B1_T10_U3_D4_data()
        data = self._numpy_to_torch(data=data, device="cpu", requires_grad=True)

        for l_buffer in [0]:
            for r_buffer in [1]:
                data["l_buffer"] = l_buffer
                data["r_buffer"] = r_buffer
                ref_costs, ref_gradients = self._compute_with_numpy_transducer(
                    data=data
                )

                self._test_costs_and_gradients(
                    data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
                )

    def _get_B1_T10_U3_D4_data(
        self,
        random=False,
        l_buffer=0,
        r_buffer=0,
        sparse=False,
        dtype=np.float32,
        nan=False,
    ):
        B, T, U, D = 2, 10, 3, 4
        data = {}
        data["logits"] = np.random.rand(B, T, U, D).astype(dtype)
        if not random:
            data["logits"].fill(0.1)
        if nan:
            for i in range(B):
                data["logits"][i][0][0][0] = np.nan
        data["src_lengths"] = np.array([10, 10], dtype=np.int32)
        data["tgt_lengths"] = np.array([2, 2], dtype=np.int32)
        data["targets"] = np.array([[1, 2], [1, 2]], dtype=np.int32)
        data["blank"] = 0
        data["wp_ends"] = np.array([[0, 2, 7], [0, 2, 7]], dtype=np.int32)

        if sparse:
            sparse_joint_enc, valid_ranges, cells_per_sample = get_sparse(
                data, data["logits"], l_buffer, r_buffer
            )
            data["logits_sparse"] = sparse_joint_enc
            data["valid_ranges"] = valid_ranges
            data["cells_per_sample"] = cells_per_sample
        return data

    @common_utils.skipIfNoCuda
    def test_rnnt_restricted_B1_T10_U3_D4_gpu(self):
        for random in [False]:
            for l_buffer in [0, 1, 10]:
                for r_buffer in [0, 1, 2, 5, 10]:
                    data = self._get_B1_T10_U3_D4_data(random=random)
                    data = self._numpy_to_torch(
                        data=data, device="cuda", requires_grad=True
                    )
                    data["l_buffer"] = l_buffer
                    data["r_buffer"] = r_buffer
                    ref_costs, ref_gradients = self._compute_with_numpy_transducer(
                        data=data
                    )
                    self._test_costs_and_gradients(
                        data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
                    )

    @common_utils.skipIfNoCuda
    def test_rnnt_sparse_B1_T10_U3_D4_gpu(self):
        for random in [False, True]:
            for l_buffer in [1, 2, 10]:
                for r_buffer in [1, 2, 5, 10]:
                    data = self._get_B1_T10_U3_D4_data(
                        random=random, l_buffer=l_buffer, r_buffer=r_buffer, sparse=True
                    )
                    data = self._numpy_to_torch(
                        data=data, device="cuda", requires_grad=True
                    )
                    data["l_buffer"] = l_buffer
                    data["r_buffer"] = r_buffer
                    ref_costs, ref_gradients = self._compute_with_numpy_transducer(
                        data=data
                    )
                    self._test_costs_and_gradients(
                        data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
                    )

    @common_utils.skipIfNoCuda
    def test_rnnt_sparse_nonfused_log_smax_gpu(self):
        for random in [False, True]:
            for l_buffer in [1, 2, 10]:
                for r_buffer in [1, 2, 5, 10]:
                    data = self._get_B1_T10_U3_D4_data(
                        random=random, l_buffer=l_buffer, r_buffer=r_buffer, sparse=True
                    )
                    data = self._numpy_to_torch(
                        data=data, device="cuda", requires_grad=True
                    )
                    data["l_buffer"] = l_buffer
                    data["r_buffer"] = r_buffer
                    data["fused_log_smax"] = False
                    ref_costs, ref_gradients = self._compute_with_numpy_transducer(
                        data=data
                    )
                    self._test_costs_and_gradients(
                        data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
                    )

    @common_utils.skipIfNoCuda
    def test_rnnt_sparse_B1_T10_U3_D4_gpu_fp16(self):
        for random in [False, True]:
            for l_buffer in [1, 2, 10]:
                for r_buffer in [1, 2, 5, 10]:
                    data = self._get_B1_T10_U3_D4_data(
                        random=random,
                        l_buffer=l_buffer,
                        r_buffer=r_buffer,
                        sparse=True,
                        dtype=np.float16,
                    )
                    data = self._numpy_to_torch(
                        data=data, device="cuda", requires_grad=True
                    )
                    data["l_buffer"] = l_buffer
                    data["r_buffer"] = r_buffer
                    ref_costs, ref_gradients = self._compute_with_numpy_transducer(
                        data=data
                    )
                    self._test_costs_and_gradients(
                        data=data,
                        ref_costs=ref_costs,
                        ref_gradients=ref_gradients,
                        atol=1e-3,
                        rtol=10,
                    )


class PyTorchTransducerLossMultipleHyposTest(unittest.TestCase):
    def _get_data_multiple_hypo(
        self, random=False, l_buffer=0, r_buffer=0, sparse=False, dtype=np.float32
    ):

        B, T, U, D, H = 2, 10, 3, 4, 3
        data = {}
        data["logits"] = np.random.rand(B * H, T, U, D).astype(dtype)
        if not random:
            data["logits"].fill(0.1)
        data["src_lengths"] = np.array([10, 10], dtype=np.int32)
        data["tgt_lengths"] = np.array([2, 2, 2, 2, 2, 2], dtype=np.int32)
        data["targets"] = np.array(
            [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]], dtype=np.int32
        )
        data["blank"] = 0
        data["wp_ends"] = np.array(
            [[0, 2, 7], [0, 2, 7], [0, 2, 7], [0, 2, 7], [0, 2, 7], [0, 2, 7]],
            dtype=np.int32,
        )
        data["nbest_wers"] = np.array([0, 1, 1, 0, 1, 1])
        if sparse:
            sparse_joint_enc, valid_ranges, cells_per_sample = get_sparse(
                data, data["logits"], l_buffer, r_buffer, H
            )
            data["logits_sparse"] = sparse_joint_enc
            data["valid_ranges"] = valid_ranges
            data["cells_per_sample"] = cells_per_sample
        return data

    @common_utils.skipIfNoCuda
    def test_rnnt_sparse_multi_hypo_gpu(self):
        for random in [False, True]:
            for l_buffer in [1, 2, 10]:
                for r_buffer in [1, 2, 5, 10]:
                    data = self._get_data_multiple_hypo(
                        random=random, l_buffer=l_buffer, r_buffer=r_buffer, sparse=True
                    )
                    data = PyTorchTransducerLossTest._numpy_to_torch(
                        data=data, device="cuda", requires_grad=True
                    )
                    data["l_buffer"] = l_buffer
                    data["r_buffer"] = r_buffer
                    (
                        ref_costs,
                        ref_gradients,
                    ) = PyTorchTransducerLossTest._compute_with_numpy_transducer(
                        data=data
                    )
                    PyTorchTransducerLossTest._test_costs_and_gradients(
                        self=self,
                        data=data,
                        ref_costs=ref_costs,
                        ref_gradients=ref_gradients,
                    )
