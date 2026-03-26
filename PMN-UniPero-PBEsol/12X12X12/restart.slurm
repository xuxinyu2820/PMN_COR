#!/bin/bash
#SBATCH --job-name=450K_restart        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --account=m5218
#SBATCH -C gpu
#SBATCH --qos=shared
#SBATCH --time=23:00:00          # total run time limit (HH:MM:SS)

module load python
conda activate /global/cfs/cdirs/m5025/pinchenx/conda_envs/dp2211
python -u 01.restart.py

### use "sbatch --dependency=afterany:******** restart.slurm" to submit a job that will be started after the job with the given job ID finishes.
