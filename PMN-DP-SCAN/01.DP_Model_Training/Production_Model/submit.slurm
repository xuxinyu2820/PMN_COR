#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 23:00:00
##SBATCH --mem=32G 
#SBATCH --gres=gpu:1
#SBATCH --job-name=train
module purge
module load anaconda3
conda activate dp-dev


if [ ! -f tag_0_finished ] ;then
  { if [ ! -f model.ckpt.index ]; then dp train input.json; else dp train input.json --restart model.ckpt; fi }  1>> train.log 2>> train.log 
  if test $? -ne 0; then exit 1; else touch tag_0_finished; fi 
fi
mv lcurve.out lcurve.out.train


if [ ! -f tag_1_finished ] ;then
  dp freeze  1>> train.log 2>> train.log 
  if test $? -ne 0; then exit 1; else touch tag_1_finished; fi 
fi

if [ ! -f tag_2_finished ] ;then
  dp compress -i frozen_model.pb -o model-compress.pb 
  # dp freeze  1>> train.log 2>> train.log 
  if test $? -ne 0; then exit 1; else touch tag_2_finished; fi 
fi

python dataset.py
