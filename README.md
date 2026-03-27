# PMN_COR

This repo is used to store the dataset, models, code and results for supporting and reproducing the paper: "Intrinsic structure of relaxor ferroelectrics from first principles".
Some large trajectory files and logging files are removed due to the size limit of GitHub. But crucial outputs are kept for immediate reproduction of main results in the paper.

_Directory Structure_

```
PMN-CACE-LR-PBEsol/  -> This directory contains the PMN structural optimization and analysis using the CACE-LR model with PBEsol functional. 

PMN-CACE-LR-SCAN/  -> This directory contains the results using the CACE-LR model with SCAN functional. 

PMN-DP-SCAN/ -> This directory contains the code and results using the Deep Potential (DP) model trained on SCAN data.

- 01.DP_Model_Training: training setup and configurations  
- 02.Structural_Optimization: Monte Carlo / relaxation results  
- 03.Molecular_Dynamics: MD simulations 

PMN-UniPero-PBEsol/ -> This directory contains the results using the UniPero universal potential (PBEsol). 

PMN-UniPero-PBEsol/ -> This directory contains the results for PST using UniPero.

PMN-UniPero-PBEsol/ -> This directory contains the results for PZT using UniPero.
```

_Reproducibility_

Softwares:

- Python 3.11.9
- DeePMD-kit 2.2.11
- ASE 3.23.0
- DPGEN
- VASP 6.2.1; POTCAR: Mg_pv 13Apr2007, Nb_pv 08Apr2002, O 08Apr2002, Pb_d 06Sep2000
- PeroStruct: https://github.com/Kehan-Cai-nanako/PeroStruc.git
- CACE-LR model (in-house MLIP, implementation will be released later when it's available).  
  A related implementation can be found at: https://github.com/BingqingCheng/cace-lr-fit

To reproduce, any v2 version of DeePMD-kit should be able to work. 
