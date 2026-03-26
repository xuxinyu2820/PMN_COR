'''
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu --account m5025
conda activate /global/cfs/cdirs/m5025/pinchenx/conda_envs/dp2211
'''
import os
import numpy as np
from ase.io import Trajectory
from ase.neighborlist import neighbor_list



# read trajectory
traj_path = "../../01.MD_Simulation/NVT/300K/pmn_300K_5000000steps.traj"
step_stride = 10
r_max = 7.0
dr = 0.01

pairs = [
    ('Mg', 'O'),
    ('Nb', 'O'),
    ('Pb', 'O'),
    ('O', 'O'),
    #('Pb', 'Mg'),
    #('Pb', 'Nb'),
    #('Pb', 'Pb'),
]

# r bins
bins = np.arange(0.0, r_max + dr, dr)
r_centers = 0.5 * (bins[:-1] + bins[1:])
shell_vol = 4.0 * np.pi * r_centers**2 * dr

# load trajectory and prepare frame indices
traj = Trajectory(traj_path, 'r')
n_total = len(traj)
frame_indices = list(range(0, n_total, step_stride))

n_frames = len(frame_indices)
n_pairs = len(pairs)
n_r = len(r_centers)

# array to hold all g(r) results
gr_all = np.zeros((n_frames, n_pairs, n_r), dtype=float)

# main loop over selected frames
for k, fidx in enumerate(frame_indices):
    print(f"Processing frame {fidx+1}/{n_total}  ({k+1}/{n_frames})")
    pmn = traj[fidx]
    pmn.set_pbc([True, True, True])

    V = pmn.get_volume()
    symbols = np.array(pmn.get_chemical_symbols())
    unique_syms, counts = np.unique(symbols, return_counts=True)
    Ns = {el: cnt for el, cnt in zip(unique_syms, counts)}

    # neighbor list
    i, j, d = neighbor_list('ijd', pmn, r_max)

    for p_idx, (a, b) in enumerate(pairs):
        if a != b:
            mask = (((symbols[i] == a) & (symbols[j] == b)) |
                    ((symbols[i] == b) & (symbols[j] == a)))
            norm_pairs = Ns.get(a, 0) * Ns.get(b, 0)
        else:
            mask = ((symbols[i] == a) & (symbols[j] == a))
            na = Ns.get(a, 0)
            norm_pairs = na * (na - 1) / 2.0

        # histogram of pairs in each r-bin
        hist, _ = np.histogram(d[mask], bins=bins)

        # ideal gas expected number of pairs (consistent with your formula)
        ideal = (norm_pairs / V) * shell_vol

        # g(r) = actual / ideal
        g = np.zeros_like(r_centers, dtype=float)
        valid = ideal > 0
        g[valid] = hist[valid] / ideal[valid]

        gr_all[k, p_idx, :] = g

# save results
out_dir = "./rdf_results"
os.makedirs(out_dir, exist_ok=True)
out_name = f"rdf_300K"
out_path = os.path.join(out_dir, out_name)

np.savez_compressed(
    out_path,
    r=r_centers,
    pairs=np.array(pairs, dtype='U3'),
    frame_indices=np.array(frame_indices, dtype=int),
    gr=gr_all,
    r_max=np.float64(r_max),
    dr=np.float64(dr),
)

print(f"Done. Saved RDF to: {out_path}")
print(f"Shape of gr: {gr_all.shape}  (frames, pairs, r_bins)")
