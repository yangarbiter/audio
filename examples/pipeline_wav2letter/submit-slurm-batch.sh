#! /bin/bash

#SBATCH --job-name=torchaudiomodel
#SBATCH --output=/checkpoint/%u/jobs/audio-%A-%a.out
#SBATCH --error=/checkpoint/%u/jobs/audio-%A-%a.err
#SBATCH --signal=USR1@600
#SBATCH --open-mode=append
#SBATCH --time=1200
#SBATCH --nodes=1
#SBATCH --array=1-1
# number of CPUs = 2x (number of data workers + number of GPUs requested)

python main.py \
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
    --epochs 30 \
	--dataset-root "/private/home/vincentqb/LibriSpeech/"
