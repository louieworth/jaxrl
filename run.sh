# mujoco: "HalfCheetah-v4" "Hopper-v4" "Walker2d-v4" "Ant-v4" "Humanoid-v4"

# "walker-run" "cheetah-run" "acrobot-swingup" "finger-turn_hard" "fish-swim" 
#  "humanoid-stand" "humanoid-run" "quadruped-run" "hopper-hop" "swimmer-swimmer15"
# for env in "${envs[@]}"; do
GPU_LIST=(0 1)
env_list=("HalfCheetah-v4" "Hopper-v4")
# declare -a envs=("HalfCheetah-v4" "Hopper-v4" "Walker2d-v4" "Ant-v4"
#  "walker-run" "cheetah-run" "acrobot-swingup" "fish-swim"  "quadruped-run" "hopper-hop"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.1
# for env in "${envs[@]}"; do
for seed in 0 1; do
    for env in ${env_list[*]}; do
    GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE 
    python examples/train.py \
    --env_name="$env" \
    --seed=$seed \
    --reset_buffer=False &

    sleep 2
    let "task=$task+1"
done
done

# ('no_prune', 'magnitude', 'random', 'saliency', 'magnitude_ste', 'random_ste', 
# 'global_magnitude', 'global_saliency', 'static_sparse', 'rigl','set')