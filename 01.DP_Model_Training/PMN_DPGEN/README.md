These folders are the results of the DP-GEN active learning for PMN.

_Folder Structure_
The folder `init.ordered` and `init.disordered` are the initial training sets.

The folder `iter.000000` to `iter.000014` are the results of the DP-GEN active learning. MD trajectories files and VASP outputs are removed from these folders due to Github size limit.

`iter.0000**/02.fp/data.***` are the final datasets for model training. 

The folder `config` contains the initial configuration files for PMN. `config/disordered` contains the initial configurations with homogeneous disorder. `config/ordered` and `config/ordered332` contains the initial configurations prescribed by the random-site model (as known as the random-layer model). 

The folder `vasp_scripts` contains the VASP input files.

_Reproducibility_
1. To reproduce the results, you need to have the DP-GEN, VASP, DeePMD-kit v2 installed. Of course, you need a license of VASP.
2. param.json and machine.json are the parameter file for the DP-GEN active learning.
3. The `vasp_scripts` folder contains the VASP input files for labeling the atomic configurations coming from the exploration stage of DP-GEN active learning.
4. To run the DP-GEN active learning, you need to run `dpgen run param.json machine.json 1> out.log 2>&1`. See the documentation of DP-GEN for more details.
5. After getting the final datasets, go to ../Production_Model to train the production model.
