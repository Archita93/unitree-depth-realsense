#!/bin/bash
#SBATCH --job-name=generate-demos
#SBATCH --output=/mnt/home/yh974/logs/generate_demos_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=1400

conda activate fasttd3

python scripts/rsl_rl/generate_demos.py \
  --task Unitree-Go2-Parkour-Record \
  --checkpoint logs/rsl_rl/unitree_go2_parkour_teacher/2025-08-11_00-15-33/model_33500.pt \
  --num_envs 100 \
  --headless \
  --num_samples 1000 \
  --enable_cameras
