#/bin/bash
## Code to run all the jobs for this codebase 

## Discrete RL Envs
strings=(
    "MinAtar/SpaceInvaders-v0"
    "MinAtar/Asterix-v0"
    "MinAtar/Breakout-v0"
    "MinAtar/Seaquest-v0"
    "MinAtar/Freeway-v0"
    "LunarLander-v2"
)
for env in "${strings[@]}"; do
    # echo "$env"
    sbatch --array=1-4 --export=ALL,ALG='cleanrl/dqn.py',ENV_ID=$env,ARGSS='--total_timesteps 25000000 --intrinsic_reward_scale=0.2' launch.sh
    sbatch --array=1-4 --export=ALL,ALG='cleanrl/ppo.py',ENV_ID=$env,ARGSS='--total_timesteps 25000000 --intrinsic_reward_scale=0.2' launch.sh

    sbatch --array=1-4 --export=ALL,ALG='cleanrl/dqn.py',ENV_ID=$env,INTRINSIC_REWARDS='--intrinsic_rewards RND',ARGSS='--total_timesteps 25000000 --intrinsic_reward_scale=0.2' launch.sh
    sbatch --array=1-4 --export=ALL,ALG='cleanrl/ppo.py',ENV_ID=$env,INTRINSIC_REWARDS='--intrinsic_rewards E3B',ARGSS='--total_timesteps 25000000 --intrinsic_reward_scale=0.2' launch.sh
done

##Continuous RL envs
strings=(
    "Walker2d-v4"
    "HalfCheetah-v4"
    "Humanoid-v4"
    "BipedalWalker-v3"
)
for env in "${strings[@]}"; do
    # echo "$env"
    # sbatch --array=1-4 --export=ALL,ALG='cleanrl/ppo_continuous_action.py',ENV_ID=$env,ARGSS='--total_timesteps 10000000' launch.sh
    sbatch --array=1-4 --export=ALL,ALG='cleanrl/ppo_continuous_action.py',ENV_ID=$env,ARGSS='--intrinsic_rewards RND --intrinsic_reward_scale=0.2 --num_envs 8 --total_timesteps 10000000' launch.sh
    sbatch --array=1-4 --export=ALL,ALG='cleanrl/ppo_continuous_action.py',ENV_ID=$env,ARGSS='--intrinsic_rewards RND --intrinsic_reward_scale=0.5 --num_envs 8 --total_timesteps 10000000' launch.sh
done

## Atari RL envs
# strings=(
#     "MontezumaRevengeNoFrameskip-v4"
#     "AsterixNoFrameskip-v4"
#     "SpaceInvadersNoFrameskip-v4"
#     "PitfallNoFrameskip-v4"
#     "BattleZoneNoFrameskip-v4"
#     "NameThisGameNoFrameskip-v4"
#     "PhoenixNoFrameskip-v4"
# )
# for env in "${strings[@]}"; do
#     # echo "$env"
#     sbatch --array=1-4 --export=ALL,ALG='cleanrl/dqn_atari.py',ENV_ID=$env,ARGSS='--total_timesteps 20000000 --intrinsic_rewards RND --intrinsic_reward_scale=0.5' launchGPU.sh
#     sbatch --array=1-4 --export=ALL,ALG='cleanrl/dqn_atari.py',ENV_ID=$env,ARGSS='--total_timesteps 20000000 --intrinsic_rewards RND --intrinsic_reward_scale=0.2' launchGPU.sh
#     sbatch --array=1-4 --export=ALL,ALG='cleanrl/ppo_atari.py',ENV_ID=$env,ARGSS='--total_timesteps 25000000 --intrinsic_rewards RND --intrinsic_reward_scale=0.5' launchGPU.sh
#     sbatch --array=1-4 --export=ALL,ALG='cleanrl/ppo_atari.py',ENV_ID=$env,ARGSS='--total_timesteps 25000000 --intrinsic_rewards RND --intrinsic_reward_scale=0.2' launchGPU.sh
# done