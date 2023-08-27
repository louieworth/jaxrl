#!/bin/bash
# Script to reproduce results
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1
export WANDB_API_KEY=9d45bb78a65fb0f3b0402a9eae36ed832ae8cbdc
GPU_LIST=(0 1 2 3)

env_list=("HalfCheetah-v4" "Hopper-v4" "Walker2d-v4" "Ant-v4")
# mujoco: "HalfCheetah-v4" "Hopper-v4" "Walker2d-v4" "Ant-v4"
# "walker-run" "cheetah-run" "acrobot-swingup" "finger-turn_hard" "fish-swim" 
#  "humanoid-stand" "humanoid-run" "quadruped-run" "hopper-hop" "swimmer-swimmer15"

# 
for seed in 3 4; do
        for env in ${env_list[*]}; do
        GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE 
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 python examples/train.py \
        --env_name="$env" \
        --seed=$seed \
        --layer_normalization=False \
        --prune_algorithm='rigl' \
        --prune_actor_sparsity=0.95 \
        --prune_critic_sparsity=0.95 \
        --prune_dist_type='erk' &

        sleep 2
        let "task=$task+1"
        done
done




# ('no_prune', 'magnitude', 'random', 'saliency', 'magnitude_ste', 'random_ste', 
# 'global_magnitude', 'global_saliency', 'static_sparse', 'rigl','set')