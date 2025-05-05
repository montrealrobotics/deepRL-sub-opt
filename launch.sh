#!/bin/bash
## Run with sbatch --array=1-3 launch.sh --export=algorithm='dqn',envID='MinAtar/SpaceInvaders-v0'

#SBATCH --partition=long-cpu                             # Ask for unkillable job
#SBATCH --cpus-per-task=4                               # Ask for 2 CPUs
#SBATCH --mem=32G                                        # Ask for 10 GB of RAM
#SBATCH --time=2:55:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/g/glen.berseth/slurm-%j.out  # Write the log on scratch

module load cudatoolkit/12.1 miniconda/3
conda activate cleanrl

echo $ALG
echo $ENV_ID

python $ALG --seed $SLURM_ARRAY_TASK_ID --env-id $ENV_ID --track $INTRINSIC_REWARDS --wandb-project-name sub-optimality
