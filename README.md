# PMN_COR

This repo is used to store the dataset, models, code and results for supporting and reproducing the paper: "Intrinsic structure of PMN relaxor from first principles".
Some large trajectory files and logging files are removed due to the size limit of GitHub. But crucial outputs are kept for immediate reproduction of main results in the paper.

_Directory Structure_

```
01.DP_Model_Training/  -> This directory contains the code and results for training the DeePMD-kit model.
02.Structural_Optimization/ -> This directory contains the code and results for structural optimization with the FIRE-Swap algorithm.
03.Molecular_Dynamics/ -> This directory contains the code and results for molecular dynamics simulations based on the optimized compositional structure.
```

_Reproducibility_

Softwares:

- Python 3.11.9
- DeePMD-kit 2.2.11
- ASE 3.23.0
- DPGEN
- VASP 6.2.1; POTCAR: Mg_pv 13Apr2007, Nb_pv 08Apr2002, O 08Apr2002, Pb_d 06Sep2000

To reproduce, any v2 version of DeePMD-kit should be able to work. 
