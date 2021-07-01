"""
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/train.py
"""
import argparse
from datetime import datetime
from functools import partial
import logging
import random
import os
from time import time

import torch
import torchaudio
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchaudio.models.tacotron2 import Tacotron2
from tqdm import tqdm

# https://github.com/NVIDIA/apex
# from apex import amp
# amp.lists.functional_overrides.FP32_FUNCS.remove('softmax')
# amp.lists.functional_overrides.FP16_FUNCS.append('softmax')

from datasets import text_mel_collate_fn, split_process_dataset, SpectralNormalization
from utils import save_checkpoint

from loss_function import Tacotron2Loss


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(os.path.basename(__file__))


def parse_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument(
        "--dataset",
        default="ljspeech",
        choices=["ljspeech"],
        type=str,
        help="select dataset to train with",
    )
    parser.add_argument(
        '-d',
        '--dataset-path',
        type=str,
        default='./',
        help='path to dataset'
    )
    parser.add_argument(
        "--val-ratio",
        default=0.1,
        type=float,
        help="the ratio of waveforms for validation"
    )

    parser.add_argument(
        '--anneal-steps',
        nargs='*',
        help='epochs after which decrease learning rate'
    )
    parser.add_argument(
        '--anneal-factor',
        type=float,
        choices=[0.1, 0.3],
        default=0.1,
        help='factor for annealing learning rate'
    )

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument(
        '--epochs',
        type=int,
        required=True,
        help='number of total epochs to run'
    )
    training.add_argument(
        '--checkpoint-path',
        type=str,
        default='',
        help='checkpoint path. If a file exists, the program will load it and resume training.'
    )
    training.add_argument(
        '--workers',
        default=8,
        type=int,
        help="number of data loading workers",
    )
    training.add_argument(
        "--validate-and-checkpoint-freq",
        default=10,
        type=int,
        metavar="N",
        help="validation frequency in epochs",
    )

    optimization = parser.add_argument_group('optimization setup')
    optimization.add_argument(
        '-lr',
        '--learning-rate',
        default=1e-3,
        type=float,
        help='learing rate'
    )
    optimization.add_argument(
        '--weight-decay',
        default=1e-6,
        type=float,
        help='weight decay'
    )
    optimization.add_argument(
        '-bs',
        '--batch-size',
        default=32,
        type=int,
        help='batch size per GPU'
    )
    optimization.add_argument(
        '--grad-clip',
        default=5.0,
        type=float,
        help='enables gradient clipping and sets maximum gradient norm value'
    )

    # dataset parameters
    dataset = parser.add_argument_group('dataset parameters')
    dataset.add_argument(
        '--text-cleaners',
        nargs='*',
        default=['english_cleaners'],
        type=str,
        help='Type of text cleaners for input text'
    )

    # model parameters
    model = parser.add_argument_group('model parameters')
    model.add_argument('--n-frames-per-step', default=1, type=int,
                       help='')
    model.add_argument('--symbols-embedding-dim', default=512, type=int,
                       help='')
    model.add_argument('--encoder-kernel-size', default=5, type=int,
                       help='')
    model.add_argument('--encoder-n-convolutions', default=3, type=int,
                       help='')
    model.add_argument('--encoder-embedding-dim', default=512, type=int,
                       help='')
    model.add_argument('--attention-rnn-dim', default=1024, type=int,
                       help='')
    model.add_argument('--attention-location-n-filters', default=32, type=int,
                       help='')
    model.add_argument('--attention-location-kernel-size', default=31, type=int,
                       help='')
    model.add_argument('--decoder-rnn-dim', default=1024, type=int,
                       help='')
    model.add_argument('--prenet-dim', default=256, type=int,
                       help='')
    model.add_argument('--max-decoder-steps', default=2000, type=int,
                       help='')
    model.add_argument('--gate-threshold', default=0.5, type=float,
                       help='')
    model.add_argument('--p-attention-dropout', default=0.1, type=float,
                       help='')
    model.add_argument('--p-decoder-dropout', default=0.1, type=float,
                       help='')
    model.add_argument('--postnet-embedding-dim', default=512, type=float,
                       help='')
    model.add_argument('--postnet-kernel-size', default=5, type=float,
                       help='')
    model.add_argument('--postnet-n-convolutions', default=5, type=float,
                       help='')
    model.add_argument('--mask-padding', action='store_true',
                       default=False, type=bool,
                       help='')
    model.add_argument('--decoder-no-early-stopping', action='store_false',
                       default=True, type=bool,
                       help='')

    # audio parameters
    audio = parser.add_argument_group('audio parameters')
    audio.add_argument('--sample-rate', default=22050, type=int, # sampling rate
                       help='Sampling rate')
    audio.add_argument('--n-fft', default=1024, type=int, # filter_length
                       help='Filter length')
    audio.add_argument('--hop-length', default=256, type=int,
                       help='Hop (stride) length')
    audio.add_argument('--win-length', default=1024, type=int,
                       help='Window length')
    audio.add_argument('--n-mels', default=80, type=int, # n-mel-channels
                       help='')
    audio.add_argument('--mel-fmin', default=0.0, type=float,
                       help='Minimum mel frequency')
    audio.add_argument('--mel-fmax', default=8000.0, type=float,
                       help='Maximum mel frequency')

    return parser


def adjust_learning_rate(epoch, optimizer, learning_rate,
                         anneal_steps, anneal_factor):
    p = 0
    if anneal_steps is not None:
        for _, a_step in enumerate(anneal_steps):
            if epoch >= int(a_step):
                p = p + 1

    if anneal_factor == 0.3:
        lr = learning_rate * ((0.1 ** (p // 2)) * (1.0 if p % 2 == 0 else 0.3))
    else:
        lr = learning_rate * (anneal_factor ** p)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def batch_to_gpu(batch):
    text_padded, input_lengths, mel_padded, gate_padded, \
        output_lengths = batch
    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    mel_padded = to_gpu(mel_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    x = (text_padded, input_lengths, mel_padded, output_lengths)
    y = (mel_padded, gate_padded)
    return x, y


def training_step(model, train_batch, batch_idx):
    (text_padded, input_lengths, mel_padded, output_lengths), y = batch_to_gpu(train_batch)
    y_pred = model(text_padded, input_lengths, mel_padded, output_lengths)
    loss = Tacotron2Loss(reduction="mean")(y_pred, y)
    return loss


def validation_step(model, val_batch, batch_idx):
    (text_padded, input_lengths, mel_padded, output_lengths), y = batch_to_gpu(val_batch)
    y_pred = model(text_padded, input_lengths, mel_padded, output_lengths)
    loss = Tacotron2Loss(reduction="sum")(y_pred, y)
    return loss


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    if rt.is_floating_point():
        rt = rt / world_size
    else:
        rt = rt // world_size
    return rt


def run(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    logger.info(f"[Rank: {rank}] in")

    torch.manual_seed(0)

    torch.cuda.set_device(rank)

    model = Tacotron2(
        mask_padding=args.mask_padding,
        n_mels=args.n_mels,
        n_symbols=148,  # len(text.symbols.symbols)
        symbols_embedding_dim=args.symbols_embedding_dim,
        encoder_kernel_size=args.encoder_kernel_size,
        encoder_n_convolutions=args.encoder_n_convolutions,
        attention_rnn_dim=args.attention_rnn_dim,
        attention_dim=args.attention_dim,
        attention_location_n_filters=args.attention_location_n_filters,
        attention_location_kernel_size=args.attention_location_kernel_size,
        n_frames_per_step=args.n_frames_per_step,
        decoder_rnn_dim=args.decoder_rnn_dim,
        prenet_dim=args.prenet_dim,
        max_decoder_steps=args.max_decoder_steps,
        gate_threshold=args.gate_threshold,
        p_attention_dropout=args.p_attention_dropout,
        p_decoder_dropout=args.p_decoder_dropout,
        postnet_embedding_dim=args.postnet_embedding_dim,
        postnet_kernel_size=args.postnet_kernel_size,
        postnet_n_convolutions=args.postnet_n_convolutions,
        decoder_no_early_stopping=args.decoder_no_early_stopping,
    ).cuda(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    best_loss = 10.0
    start_epoch = 0

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        logger.info(f"Checkpoint: loading '{args.checkpoint_path}'")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(args.checkpoint_path, map_location=map_location)

        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        logger.info(
            f"Checkpoint: loaded '{args.checkpoint_path}' at epoch {checkpoint['epoch']}"
        )

    transforms = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            n_fft=args.n_fft,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            f_max=args.mel_fmax,
            f_min=args.mel_fmin,
            mel_scale='slaney',
            power=1,
            norm='slaney',
            normalized=False,
            hop_length=args.hop_length,
            win_length=args.win_length,
        ),
        SpectralNormalization()
    )
    trainset, valset = split_process_dataset(
        args.dataset, args.dataset_path, args.val_ratio, transforms)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        shuffle=True,
        num_replicas=world_size,
        rank=rank,
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        valset,
        shuffle=False,
        num_replicas=world_size,
        rank=rank,
    )

    loader_params = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "shuffle": False,
        "pin_memory": False,
        "drop_last": False,
        "collate_fn": partial(text_mel_collate_fn, n_frames_per_step=args.n_frames_per_step),
    }

    train_loader = DataLoader(trainset, sampler=train_sampler, **loader_params)
    val_loader = DataLoader(valset, sampler=val_sampler, **loader_params)
    dist.barrier()

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"[rank: {rank}, Epoch: {epoch}] start")
        start = time()

        model.train()
        trn_loss, counts = 0, 0

        if rank == 0:
            iterator = tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader))
        else:
            iterator = enumerate(train_loader)
        for i, batch in iterator:
            if 'CUBLAS_WORKSPACE_CONFIG' in os.environ and i == 2:
                break
            adjust_learning_rate(epoch, optimizer, args.learning_rate,
                                 args.anneal_steps, args.anneal_factor)

            model.zero_grad()

            loss = training_step(model, batch, i)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip)

            optimizer.step()

            trn_loss += loss.item() * len(batch[0])
            counts += len(batch[0])

        logger.info(f"[Rank: {rank}, Epoch: {epoch}] time: {time()-start}; trn_loss: {trn_loss/counts}")

        if ((epoch + 1) % args.validate_and_checkpoint_freq == 0) or (epoch == args.epochs - 1):

            if 'CUBLAS_WORKSPACE_CONFIG' in os.environ:
                break

            val_start_time = time()
            model.eval()
            with torch.no_grad():
                val_loss, counts = 0, 0
                iterator = tqdm(enumerate(val_loader), desc=f"[Eval Epoch: {epoch}]", total=len(val_loader))
                for val_batch_idx, val_batch in iterator:
                    val_loss = val_loss + validation_step(model, val_batch, val_batch_idx).item()
                    counts = counts + len(val_batch[0])
                val_loss = val_loss / counts

            logger.info(f"[Epoch: {epoch}; Eval] time: {time()-val_start_time}; val_loss: {val_loss}")

            if rank == 0:
                is_best = val_loss < best_loss
                best_loss = min(val_loss, best_loss)
                logger.info(f"[Rank: {rank}, Epoch: {epoch}] Saving checkpoint to {args.checkpoint_path}")
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "best_loss": best_loss,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best,
                    args.checkpoint_path,
                )

    dist.destroy_process_group()
    if rank == 0:
        return model


def main(args):
    logger.info("Start time: {}".format(str(datetime.now())))

    torch.manual_seed(0)
    random.seed(0)

    if 'CUBLAS_WORKSPACE_CONFIG' in os.environ:
        torch.use_deterministic_algorithms(True)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '17778'

    #if args.jit:
    #    model = torch.jit.script(model)

    device_counts = torch.cuda.device_count()

    logger.info(f"# available GPUs: {device_counts}")

    if device_counts == 1:
        tacotron2 = run(0, 1, args)
    else:
        tacotron2 = mp.spawn(run, args=(device_counts, args, ), nprocs=device_counts, join=True)

    if 'CUBLAS_WORKSPACE_CONFIG' in os.environ:
        from numpy.testing import assert_equal
        # [Rank: 0, Epoch: 0] time: 6.493569612503052; trn_loss: 1394.7091064453125
        baseline_state_dict = torch.load("./test_ckpt.pth")['state_dict']
        state_dict = tacotron2.state_dict()
        for k, v in state_dict.items():
            assert_equal(v.cpu().numpy(), baseline_state_dict[k].cpu().numpy())

    print("[train] Correct!")
    logger.info(f"End time: {datetime.now()}")
    exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Tacotron 2 Training')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    main(args)
