#!/bin/bash
#SBATCH --job-name=fasttd3-ablation
#SBATCH --output=logs/fasttd3-ablation/%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=4320
#SBATCH --mem=10000M
#SBATCH --cpus-per-gpu=6


conda activate fasttd3

set -e

name="ablation"
project="locomotion-ablation-test"
num_seeds=3
fasttd3_num_steps=1500
ppo_num_steps=101

echo "Training Teacher PPO policy"

for ((seed = 0; i < num_seeds; ++i)); do
  python scripts/rsl_rl/train.py \
    --headless \
    --task Unitree-Go2-Parkour-Teacher \
    --logger wandb \
    --run_name teacher-ppo-$name \
    --log_project_name $project \
    --max_iterations $ppo_num_steps \
    --seed $seed
done

# obtain path to model weights
model_path="logs/rsl_rl/unitree_go2_parkour_teacher/$(ls logs/rsl_rl/unitree_go2_parkour_teacher/ | tail -n 1)/model_100.pt"
echo "Loading model weights from ${model_path}"

echo "Generate Demos"

python scripts/rsl_rl/generate_demos.py \
  --task Unitree-Go2-Parkour-Record-Play \
  --checkpoint ${model_path} \
  --out ./stored_transitions_ppo_teacher_$name \
  --num_envs 256 \
  --headless \
  --num_samples 8000

cd external/FastTD3/
echo "Training Teacher FastTD3 policy"

for ((seed = 0; i < num_seeds; ++i)); do
  python scripts/train_rnn.py \
    --env_name Unitree-Go2-Parkour-Teacher \
    --exp_name teacher-fasttd3-$name \
    --project $project \
    --agent fasttd3 \
    --render_interval 0 \
    --total-timesteps $fasttd3_num_steps \
    --no-use-vision \
    --seed $seed
done

echo "Training blind unprivlieged policy"

for ((seed = 0; i < num_seeds; ++i)); do
  python scripts/train_rnn.py \
    --env_name Unitree-Go2-Parkour-Student \
    --exp_name teacher-fasttd3-$name \
    --project $project \
    --agent fasttd3 \
    --render_interval 0 \
    --total-timesteps $fasttd3_num_steps \
    --no-use-vision \
    --seed $seed
done

echo "Training unprivlieged policy"

for ((seed = 0; i < num_seeds; ++i)); do
  python scripts/train_rnn.py \
    --env_name Unitree-Go2-Parkour-Student \
    --exp_name teacher-fasttd3-$name \
    --project $project \
    --agent fasttd3 \
    --render_interval 0 \
    --total-timesteps $fasttd3_num_steps \
    --use-auxiliary-loss \
    --auxiliary-loss-weight 0.01 \
    --seed $seed
done

echo "Train Student policies"

for ((seed = 0; i < num_seeds; ++i)); do
  python scripts/train_rnn.py \
    --env_name Unitree-Go2-Parkour-Student \
    --exp_name student-policy-$name \
    --project $project \
    --agent fasttd3 \
    --render_interval 0 \
    --teacher-obs-path ../../stored_transitions_ppo_teacher_$name \
    --total-timesteps $fasttd3_num_steps \
    --normalize-vision-scalar 1.0 \
    --use-auxiliary-loss \
    --auxiliary-loss-weight 0.01 \
    --seed 2
end

echo "Train blind Student policies"

for ((seed = 0; i < num_seeds; ++i)); do
  python scripts/train_rnn.py \
    --env_name Unitree-Go2-Parkour-Student \
    --exp_name student-policy-$name \
    --project $project \
    --agent fasttd3 \
    --render_interval 0 \
    --teacher-obs-path ../../stored_transitions_ppo_teacher_$name \
    --total-timesteps $fasttd3_num_steps \
    --no-use-vision \
    --seed 2
end
