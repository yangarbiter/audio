"""
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/train.py
"""
import argparse
from datetime import datetime
import logging
import os

import torch
import torchaudio
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchaudio.models.tacotron2 import Tacotron2

# https://github.com/NVIDIA/apex
#from apex import amp
#amp.lists.functional_overrides.FP32_FUNCS.remove('softmax')
#amp.lists.functional_overrides.FP16_FUNCS.append('softmax')

from datasets import TextMelCollate, split_process_dataset, TextMelLoader
from utils import save_checkpoint

from loss_function import Tacotron2Loss
from processing import NormalizeDB


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.basename(__file__))


def parse_args(parser):
    """
    Parse commandline arguments.
    """

    #parser.add_argument('-o', '--output', type=str, required=True,
    #                    help='Directory to save checkpoints')
    parser.add_argument(
        "--dataset",
        default="ljspeech",
        choices=["ljspeech", "libritts"],
        type=str,
        help="select dataset to train with",
    )
    parser.add_argument('-d', '--dataset-path', type=str,
                        default='./', help='Path to dataset')
    parser.add_argument('--log-file', type=str, default='log.json',
                        help='Filename for logging')
    parser.add_argument(
        "--val-ratio",
        default=0.1,
        type=float,
        help="the ratio of waveforms for validation",
    )

    parser.add_argument('--anneal-steps', nargs='*',
                        help='Epochs after which decrease learning rate')
    parser.add_argument('--anneal-factor', type=float, choices=[0.1, 0.3], default=0.1,
                        help='Factor for annealing learning rate')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, required=True,
                          help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=50,
                          help='Number of epochs per checkpoint')
    training.add_argument('--checkpoint-path', type=str, default='',
                          help='Checkpoint path to resume training')
    training.add_argument('--resume-from-last', action='store_true',
                          help='Resumes training from the last checkpoint; uses the directory provided with \'--output\' option to search for the checkpoint \"checkpoint_<model_name>_last.pt\"')
    training.add_argument('--dynamic-loss-scaling', type=bool, default=True,
                          help='Enable dynamic loss scaling')
    #training.add_argument('--amp', action='store_true',
    #                      help='Enable AMP')
    training.add_argument('--workers', default=8, type=int, help='')
    training.add_argument(
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency in epochs",
    )
    training.add_argument('--cudnn-enabled', action='store_true',
                          help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', action='store_true',
                          help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true',
                          help='disable uniform initialization of batchnorm layer weight')

    optimization = parser.add_argument_group('optimization setup')
    optimization.add_argument('--use-saved-learning-rate', default=False, type=bool)
    optimization.add_argument('-lr', '--learning-rate', type=float, required=True,
                              help='Learing rate')
    optimization.add_argument('--weight-decay', default=1e-6, type=float,
                              help='Weight decay')
    optimization.add_argument('--grad-clip-thresh', default=1.0, type=float,
                              help='Clip threshold for gradients')
    optimization.add_argument('-bs', '--batch-size', type=int, required=True,
                              help='Batch size per GPU')
    optimization.add_argument('--grad-clip', default=5.0, type=float,
                              help='Enables gradient clipping and sets maximum gradient norm value')

    # dataset parameters
    dataset = parser.add_argument_group('dataset parameters')
    dataset.add_argument('--load-mel-from-disk', action='store_true',
                         help='Loads mel spectrograms from disk instead of computing them on the fly')
    dataset.add_argument('--training-files',
                         default='filelists/ljs_audio_text_train_filelist.txt',
                         type=str, help='Path to training filelist')
    dataset.add_argument('--validation-files',
                         default='filelists/ljs_audio_text_val_filelist.txt',
                         type=str, help='Path to validation filelist')
    dataset.add_argument('--text-cleaners', nargs='*',
                         default=['english_cleaners'], type=str,
                         help='Type of text cleaners for input text')

    # audio parameters
    audio = parser.add_argument_group('audio parameters')
    #audio.add_argument('--max-wav-value', default=32768.0, type=float,
    #                   help='Maximum audiowave value')
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
                p = p+1

    if anneal_factor == 0.3:
        lr = learning_rate*((0.1 ** (p//2))*(1.0 if p % 2 == 0 else 0.3))
    else:
        lr = learning_rate*(anneal_factor ** p)

    #if optimizer.param_groups[0]['lr'] != lr:
    #    logging.info(step=(epoch, iteration), data={'learning_rate changed': str(optimizer.param_groups[0]['lr'])+" -> "+str(lr)})

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
    max_len = torch.max(input_lengths.data).item()
    mel_padded = to_gpu(mel_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    x = (text_padded, input_lengths, mel_padded, max_len, output_lengths)
    y = (mel_padded, gate_padded)
    return x, y


def training_step(model, train_batch, batch_idx):
    x, y = batch_to_gpu(train_batch)
    y_pred = model(x)
    loss = Tacotron2Loss()(y_pred, y)
    return loss


def validation_step(model, val_batch, batch_idx):
    x, y = batch_to_gpu(val_batch)
    y_pred = model(x)
    loss = Tacotron2Loss(reduction="sum")(y_pred, y)
    return loss


def run(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    logger.info(f"[rank: {rank}] in")

    torch.manual_seed(0)

    torch.cuda.set_device(rank)
    model = Tacotron2().cuda(rank)
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

    class SpectralNormalization(torch.nn.Module):
        def forward(self, input):
            return torch.log(torch.clamp(input, min=1e-5))

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

    #trainset_old = TextMelLoader(args.dataset_path, args.training_files, args)
    #valset = TextMelLoader(args.dataset_path, args.validation_files, args)

    #print(trainset[3025])
    #print(trainset_old[931])

    import ipdb; ipdb.set_trace()

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        num_replicas=world_size,
        rank=rank
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        valset,
        num_replicas=world_size,
        rank=rank
    )


    collate_fn = TextMelCollate(n_frames_per_step=1)
    loader_params = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": False,
        "drop_last": False,
        "collate_fn": collate_fn,
    }

    train_loader = DataLoader(trainset, shuffle=False, sampler=train_sampler, **loader_params)
    val_loader = DataLoader(valset, shuffle=False, sampler=val_sampler, **loader_params)
    if world_size > 1:
        dist.barrier()

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"[rank: {rank}] Epoch: {epoch}")
        model.train()
        trn_loss, counts = 0, 0
        for i, batch in enumerate(train_loader):
            adjust_learning_rate(epoch, optimizer, args.learning_rate,
                                 args.anneal_steps, args.anneal_factor)

            model.zero_grad()

            loss = training_step(model, batch, i)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip_thresh)

            optimizer.step()

            trn_loss += loss.item() * len(batch[0])
            counts += len(batch[0])

        logger.info(f"[Epoch: {epoch}] trn_loss: {trn_loss}")

        if rank == 0 and (not (epoch + 1) % args.print_freq or epoch == args.epochs - 1):
            model.eval()
            val_loss, counts = 0, 0
            for val_batch_idx, val_batch in enumerate(val_loader):
                val_loss += validation_step(model, val_batch, val_batch_idx).item()
                counts += len(val_batch[0])
            val_loss /= counts

            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            logger.info(f"[Rank: {rank, }Epoch: {epoch}] Saving checkpoint to {args.checkpoint_path}")
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

            logger.info(f"[Epoch: {epoch}] val_loss: {val_loss}")
    
    dist.destroy_process_group()

def main(args):
    logger.info("Start time: {}".format(str(datetime.now())))

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '17778'

    #if args.jit:
    #    model = torch.jit.script(model)

    device_counts = torch.cuda.device_count()

    logger.info(f"# available GPUs: {device_counts}")

    if device_counts == 1:
        run(0, 1, args)
    else:
        mp.spawn(run, args=(device_counts, args, ), nprocs=device_counts)

    logger.info(f"End time: {datetime.now()}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Tacotron 2 Training')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    main(args)