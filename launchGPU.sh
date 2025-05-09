#!/bin/bash
## Run with sbatch --array=1-3 launch.sh

#SBATCH --partition=long                                # Ask for unkillable job
#SBATCH --cpus-per-task=4                               # Ask for 2 CPUs
#SBATCH --ntasks=3                                      # Ask for 2 tasks per GPU 
#SBATCH --gres=gpu:1                                    # Ask for GPUs
#SBATCH --mem=96G                                       # Ask for 10 GB of RAM
#SBATCH --time=5:55:00                                  # The job will run for 3 hours
#SBATCH --tres-per-task=gpu:1                            # The job name
#SBATCH -o /network/scratch/g/glen.berseth/slurm-%j.out  # Write the log on scratch

module load cudatoolkit/12.1 miniconda/3
conda activate cleanrl

echo $ALG
echo $ENV_ID

## srun is needed to run the multiple tasks per GPU
srun python $ALG --env-id $ENV_ID --track $INTRINSIC_REWARDS --wandb-project-name sub-optimality
