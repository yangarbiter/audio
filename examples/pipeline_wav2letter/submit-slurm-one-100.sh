#! /bin/bash

#SBATCH --job-name=torchaudiomodel
#SBATCH --output=/checkpoint/%u/jobs/audio-%j.out
#SBATCH --error=/checkpoint/%u/jobs/audio-%j.err
#SBATCH --signal=USR1@600
#SBATCH --open-mode=append
#SBATCH --partition=learnfair
#SBATCH --time=4320
#SBATCH --mem-per-cpu=5120
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=80
# 2x (number of data workers + number of GPUs requested)

# PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# The ENV below are only used in distributed training with env:// initialization
export MASTER_ADDR=${SLURM_JOB_NODELIST:0:9}${SLURM_JOB_NODELIST:10:4}
export MASTER_PORT=29500

# srun --label \
#     python /private/home/vincentqb/experiment/PipelineTrain.py \
# 	--arch $arch --batch-size $bs --learning-rate $lr \
# 	--resume /private/home/vincentqb/experiment/checkpoint-$SLURM_JOB_ID-$arch-$bs-$lr.pth.tar
# 	# --distributed --world-size $SLURM_JOB_NUM_NODES --dist-url 'env://' --dist-backend='nccl'

    # --dataset-train train-clean-100 train-clean-360 train-other-500 \
CMD="python /private/home/vincentqb/audio/examples/pipeline_wav2letter/main.py \
    --reduce-lr-valid \
    --dataset-train train-clean-100 \
    --dataset-valid dev-clean \
    --batch-size 128 \
    --learning-rate .6 \
    --momentum .8 \
    --weight-decay .00001 \
    --clip-grad 0. \
    --gamma .99 \
    --hop-length 160 \
    --win-length 400 \
    --n-bins 13 \
    --normalize \
    --optimizer adadelta \
    --scheduler reduceonplateau \
    --epochs 1000 \
    --dataset-root /datasets01/librispeech/ \
    --dataset-folder-in-archive 062419 \
    --checkpoint /checkpoint/vincentqb/checkpoint/checkpoint-$SLURM_JOB_ID.pth.tar"
# CMD="$CMD --distributed --world-size $SLURM_JOB_NUM_NODES --dist-url 'env://' --dist-backend='nccl'"

>&2 echo $CMD
eval $CMD
