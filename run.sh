#!/bin/bash

#SBATCH --account acount
#SBATCH --partition partition
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=1G
#SBATCH --mail-user=email
#SBATCH --mail-type=ALL

nvidia-smi
python app.py
