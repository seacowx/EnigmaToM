#!/bin/bash

#SBATCH --job-name=ATOMIC-Aug
#SBATCH --partition=he_dev_gpu
#SBATCH --gpus=1
#SBATCH --constraint a100
#SBATCH --cpus-per-task=1
#SBATCH --mem=80GB
#SBATCH --open-mode append
#SBATCH -o atomic_aug.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hainiu.xu@kcl.ac.uk

python fill_in_blank.py
