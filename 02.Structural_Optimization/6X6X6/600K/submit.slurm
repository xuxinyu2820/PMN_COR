#!/bin/bash
#SBATCH --job-name=pmn_dislc        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --account=m5025
#SBATCH -C gpu
#SBATCH --qos=shared
#SBATCH --time=6:00:00          # total run time limit (HH:MM:SS)

## for debugging run, set --time=00:29:00  and --qos=debug.  The debug queue has a much higher priority than regular queue.
module load python
conda activate /global/cfs/cdirs/m5025/pinchenx/conda_envs/dp2211
cd /global/homes/x/xinyuxu/m5025/Ferroic/PMN/Finite_Temp_MC/6X6X6
python -u optimize.py
