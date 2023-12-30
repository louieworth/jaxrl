#!/bin/bash
# Script to reproduce results
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1
export WANDB_API_KEY=9d45bb78a65fb0f3b0402a9eae36ed832ae8cbdc
GPU_LIST=(0 1)

env_list=("HalfCheetah-v4" "Hopper-v4" "Walker2d-v4" "Ant-v4")
# env_list=("HalfCheetah-v4")
# algs=('random' 'magnitude' 'static_sparse' 'set' 'rigl')
algs=('magnitude')

# "walker-run" "cheetah-run" "acrobot-swingup" "finger-turn_hard" "fish-swim" 
#  "humanoid-stand" "humanoid-run" "quadruped-run" "hopper-hop" "swimmer-swimmer15"
for env in ${env_list[*]}; do
        for seed in 0; do
                for alg in ${algs[*]}; do
                GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
                CUDA_VISIBLE_DEVICES=$GPU_DEVICE 
                python examples/train.py \
                --env_name="$env" \
                --seed=$seed \
                --prune_algorithm="$alg" \
                --prune_actor_sparsity=0.9 \
                --prune_critic_sparsity=0.9 \
                --prune_dist_type='erk' &

                sleep 2
                let "task=$task+1"
                done
        done
done

# ('no_prune', 'random', 'magnitude',  
# 'static_sparse', 'rigl','set')


# ('no_prune', 'magnitude', 'random', 'saliency', 'magnitude_ste', 'random_ste', 
# 'global_magnitude', 'global_saliency', 'static_sparse', 'rigl','set')