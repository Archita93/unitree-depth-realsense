#!/bin/bash
#SBATCH --job-name=ppo-teacher-training
#SBATCH --output=/mnt/home/yh974/logs/ppo_teacher_training_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=1400

conda activate fasttd3

python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Parkour-Teacher --logger wandb
