#/bin/bash
## Code to run all the jobs for this codebase 

# sbatch --array=1-3 --export=ALL,ALG='cleanrl/dqn.py',ENV_ID='MinAtar/SpaceInvaders-v0' launch.sh
# sbatch --array=1-3 --export=ALL,ALG='cleanrl/dqn.py',ENV_ID='MinAtar/Asterix-v0' launch.sh
# sbatch --array=1-3 --export=ALL,ALG='cleanrl/dqn.py',ENV_ID='LunarLander-v2' launch.sh
# sbatch --array=1-3 --export=ALL,ALG='cleanrl/dqn.py',ENV_ID='MinAtar/SpaceInvaders-v0',INTRINSIC_REWARDS='--intrinsic_rewards' launch.sh
# sbatch --array=1-3 --export=ALL,ALG='cleanrl/dqn.py',ENV_ID='MinAtar/Asterix-v0',INTRINSIC_REWARDS='--intrinsic_rewards' launch.sh
# sbatch --array=1-3 --export=ALL,ALG='cleanrl/dqn.py',ENV_ID='LunarLander-v2',INTRINSIC_REWARDS='--intrinsic_rewards' launch.sh
# sbatch --array=1-3 --export=ALL,ALG='cleanrl/ppo.py',ENV_ID='MinAtar/SpaceInvaders-v0' launch.sh
# sbatch --array=1-3 --export=ALL,ALG='cleanrl/ppo.py',ENV_ID='MinAtar/Asterix-v0' launch.sh
# sbatch --array=1-3 --export=ALL,ALG='cleanrl/ppo.py',ENV_ID='LunarLander-v2' launch.sh
# sbatch --array=1-3 --export=ALL,ALG='cleanrl/ppo.py',ENV_ID='MinAtar/SpaceInvaders-v0',INTRINSIC_REWARDS='--intrinsic_rewards' launch.sh
# sbatch --array=1-3 --export=ALL,ALG='cleanrl/ppo.py',ENV_ID='MinAtar/Asterix-v0',INTRINSIC_REWARDS='--intrinsic_rewards' launch.sh
# sbatch --array=1-3 --export=ALL,ALG='cleanrl/ppo.py',ENV_ID='LunarLander-v2',INTRINSIC_REWARDS='--intrinsic_rewards' launch.sh

# sbatch --array=1-3 --export=ALL,ALG='cleanrl/ppo_continuous_action.py',ENV_ID='Walker2d-v4' launch.sh
# sbatch --array=1-3 --export=ALL,ALG='cleanrl/ppo_continuous_action.py',ENV_ID='HalfCheetah-v4' launch.sh

## For GPU jobs we can fit 3 jobs per GPU
# sbatch --array=1-2 --export=ALL,ALG='cleanrl/dqn_atari.py',ENV_ID='SpaceInvadersNoFrameskip-v4' launchGPU.sh
# sbatch --array=1-2 --export=ALL,ALG='cleanrl/dqn_atari.py',ENV_ID='AsterixNoFrameskip-v4' launchGPU.sh
sbatch --array=1-2 --export=ALL,ALG='cleanrl/dqn_atari.py',ENV_ID='MontezumaRevengeNoFrameskip-v4' launchGPU.sh
# sbatch --array=1-2 --export=ALL,ALG='cleanrl/dqn_atari.py',ENV_ID='SpaceInvadersNoFrameskip-v4',INTRINSIC_REWARDS='--intrinsic_rewards' launchGPU.sh
# sbatch --array=1-2 --export=ALL,ALG='cleanrl/dqn_atari.py',ENV_ID='AsterixNoFrameskip-v4',INTRINSIC_REWARDS='--intrinsic_rewards' launchGPU.sh
sbatch --array=1-2 --export=ALL,ALG='cleanrl/dqn_atari.py',ENV_ID='MontezumaRevengeNoFrameskip-v4',INTRINSIC_REWARDS='--intrinsic_rewards' launchGPU.sh

# sbatch --array=1-2 --export=ALL,ALG='cleanrl/ppo_atari.py',ENV_ID='SpaceInvadersNoFrameskip-v4' launchGPU.sh
# sbatch --array=1-2 --export=ALL,ALG='cleanrl/ppo_atari.py',ENV_ID='AsterixNoFrameskip-v4' launchGPU.sh
sbatch --array=1-2 --export=ALL,ALG='cleanrl/ppo_atari.py',ENV_ID='MontezumaRevengeNoFrameskip-v4' launchGPU.sh
# sbatch --array=1-2 --export=ALL,ALG='cleanrl/ppo_atari.py',ENV_ID='SpaceInvadersNoFrameskip-v4',INTRINSIC_REWARDS='--intrinsic_rewards' launchGPU.sh
# sbatch --array=1-2 --export=ALL,ALG='cleanrl/ppo_atari.py',ENV_ID='AsterixNoFrameskip-v4',INTRINSIC_REWARDS='--intrinsic_rewards' launchGPU.sh
sbatch --array=1-2 --export=ALL,ALG='cleanrl/ppo_atari.py',ENV_ID='MontezumaRevengeNoFrameskip-v4',INTRINSIC_REWARDS='--intrinsic_rewards' launchGPU.sh

