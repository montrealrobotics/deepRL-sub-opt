#!/bin/bash
## Run with sbatch --array=1-3 launch.sh

#SBATCH --partition=long                             # Ask for unkillable job
#SBATCH --cpus-per-task=4                               # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                    # Ask for GPUs
#SBATCH --mem=32G                                        # Ask for 10 GB of RAM
#SBATCH --time=2:55:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/g/glen.berseth/slurm-%j.out  # Write the log on scratch

module load cudatoolkit/12.1 miniconda/3
conda activate cleanrl

python cleanrl/dqn.py --seed $SLURM_ARRAY_TASK_ID --env-id MinAtar/SpaceInvaders-v0 --track --wandb-project-name sub-optimality
python cleanrl/dqn.py --seed $SLURM_ARRAY_TASK_ID --env-id MinAtar/SpaceInvaders-v0 --track --wandb-project-name sub-optimality --intrinsic_rewards
# python cleanrl/dqn.py --seed $SLURM_ARRAY_TASK_ID --env-id MinAtar/Asterix-v0 --track --wandb-project-name sub-optimality
# python cleanrl/dqn.py --seed $SLURM_ARRAY_TASK_ID --env-id LunarLander-v2 --track --wandb-project-name sub-optimality

# python cleanrl/ppo.py --seed $SLURM_ARRAY_TASK_ID --env-id MinAtar/SpaceInvaders-v0 --track --wandb-project-name sub-optimality
# python cleanrl/ppo.py --seed $SLURM_ARRAY_TASK_ID --env-id MinAtar/Asterix-v0 --track --wandb-project-name sub-optimality
# python cleanrl/ppo.py --seed $SLURM_ARRAY_TASK_ID --env-id LunarLander-v2 --track --wandb-project-name sub-optimality

# python cleanrl/ppo_continuous_action.py --seed $SLURM_ARRAY_TASK_ID --env-id Walker2d-v4 --track --wandb-project-name sub-optimality
# python cleanrl/ppo_continuous_action.py --seed $SLURM_ARRAY_TASK_ID --env-id HalfCheetah-v4 --track --wandb-project-name sub-optimality
