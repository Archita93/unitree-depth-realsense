#!/bin/bash
#SBATCH --job-name=train_teacher_student_policy
#SBATCH --output=logs/train_teacher_student_policy/%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=1800
#SBATCH --mem=10000M
#SBATCH --cpus-per-gpu=6


conda activate fasttd3

set -e

echo "Training Teacher PPO policy"

python scripts/rsl_rl/train.py \
  --headless \
  --task Unitree-Go2-Parkour-Teacher \
  --logger wandb \
  --run_name teacher-ppo-real-parkour \
  --log_project_name locomotion-student-teacher \
  --max_iterations 4001

# obtain path to model weights
model_path="logs/rsl_rl/unitree_go2_parkour_teacher/$(ls logs/rsl_rl/unitree_go2_parkour_teacher/ | tail -n 1)/model_4000.pt"
echo "Loading model weights from ${model_path}"

echo "Generate Demos"

python scripts/rsl_rl/generate_demos.py \
  --task Unitree-Go2-Parkour-Record-Play \
  --checkpoint ${model_path} \
  --out ./stored_transitions_ppo_teacher_real_parkour \
  --num_envs 256 \
  --headless \
  --num_samples 8000

echo "Train Student"
cd external/FastTD3/

python scripts/train_rnn.py \
  --env_name Unitree-Go2-Parkour-Student \
  --exp_name student-policy-real-parkour \
  --project locomotion-student-teacher \
  --agent fasttd3 \
  --render_interval 0 \
  --teacher-obs-path ../../stored_transitions_ppo_teacher_real_parkour \
  --total-timesteps 25000 \
  --seed 1
