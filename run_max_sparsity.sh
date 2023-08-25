# mujoco: "HalfCheetah-v4" "Hopper-v4" "Walker2d-v4" "Ant-v4" "Humanoid-v4"

# "walker-run" "cheetah-run" "acrobot-swingup" "finger-turn_hard" "fish-swim" 
#  "humanoid-stand" "humanoid-run" "quadruped-run" "hopper-hop" "swimmer-swimmer15"
# for env in "${envs[@]}"; do

# declare -a envs=("cheetah-run" "finger-turn_hard" "humanoid-run" "hopper-hop"
seeds=1
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.15
# for env in "${envs[@]}"; do
for seed in $(seq 0 $((seeds-1))); do
    CUDA_VISIBLE_DEVICES=1 python examples/train_max_sparsity.py \
    --env_name="cheetah-run" \
    --seed=$seed \
    --gate_ratio=0.1
done
# done

# ('no_prune', 'magnitude', 'random', 'saliency', 'magnitude_ste', 'random_ste', 
# 'global_magnitude', 'global_saliency', 'static_sparse', 'rigl','set')