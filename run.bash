# mujoco: "HalfCheetah-v4" "Hopper-v4" "Walker2d-v4" "Ant-v4" "Humanoid-v4" "HumanoidStandup-v4"
# "walker-run" "cheetah-run" "acrobot-swingup" "finger-turn_hard" "fish-swim" 
#  "humanoid-stand" "humanoid-run" "quadruped-run" "hopper-hop" "swimmer-swimmer15"
# for env in "${envs[@]}"; do

declare -a envs=("HalfCheetah-v4" "Hopper-v4" "Walker2d-v4" "Ant-v4")
seeds=1
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.15
for env in "${envs[@]}"; do
for seed in $(seq 0 $((seeds-1))); do
    CUDA_VISIBLE_DEVICES=0 python examples/train.py \
    --env_name="Hopper-v4" \
    --seed=$seed
done
done

# ('no_prune', 'magnitude', 'random', 'saliency', 'magnitude_ste', 'random_ste', 
# 'global_magnitude', 'global_saliency', 'static_sparse', 'rigl','set')