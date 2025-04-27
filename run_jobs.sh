#/bin/bash
## Code to run all the jobs for this codebase 

sbatch --array=1-3 launch.sh --export=algorithm='dqn',envID='MinAtar/SpaceInvaders-v0'
sbatch --array=1-3 launch.sh --export=algorithm='dqn',envID='MinAtar/Asterix-v0'
sbatch --array=1-3 launch.sh --export=algorithm='dqn',envID='LunarLander-v2'
sbatch --array=1-3 launch.sh --export=algorithm='ppo',envID='MinAtar/SpaceInvaders-v0'
sbatch --array=1-3 launch.sh --export=algorithm='ppo',envID='MinAtar/Asterix-v0'
sbatch --array=1-3 launch.sh --export=algorithm='ppo',envID='LunarLander-v2'

# sbatch --array=1-3 launch.sh --export=algorithm='ppo_continuous_action',envID='Walker2d-v4'
# sbatch --array=1-3 launch.sh --export=algorithm='ppo_continuous_action',envID='HalfCheetah-v4'

# sbatch --array=1-3 launchGPU.sh --export=algorithm='dqn_atari',envID='SpaceInvadersNoFrameskip-v4'
# sbatch --array=1-3 launchGPU.sh --export=algorithm='dqn_atari',envID='AsterixNoFrameskip-v4'
# sbatch --array=1-3 launchGPU.sh --export=algorithm='dqn_atari',envID='MontezumaRevengeNoFrameskip-v4'