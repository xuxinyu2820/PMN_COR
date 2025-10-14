This folder contains the training scripts for the production model.

`input.json` is the input file for the training.

`submit.slurm` is the SLURM script for the training. You can follow it to reproduce the model.

`dataset.py` is the script for the testing. `error.png` is the outcome. 

The deep potential model without compression is `frozen_model.pb`. It is slow and should not be used for production.
The compressed deep potential model is `model-compress.pb`. It is fast and should be used for production.

_Reproducibility_

1. To reproduce the results, you need to have the DeePMD-kit v2 installed.
2. CPU is enough for the training. But GPU will speed up the training.
3. Follow `submit.slurm` script to train and compress the model.