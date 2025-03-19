#!/bin/bash

#!/bin/bash


#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=16G                                        # Ask for 10 GB of RAM
#SBATCH --time=2:00:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/g/glen.berseth/slurm-%j.out  # Write the log on scratch


module load cudatoolkit/12.1 miniconda/3
conda activate cleanrl

python cleanrl/dqn.py --seed 1 --env-id MinAtar/SpaceInvaders-v0 --track --wandb-project-name sub-optimality
python cleanrl/dqn.py --seed 2 --env-id MinAtar/SpaceInvaders-v0 --track --wandb-project-name sub-optimality