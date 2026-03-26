import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.linewidth'] = 3     
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20

# load data
data = np.load("./rdf_results/rdf_300K_homo.npz")

r = data["r"]                            # (n_bins,)
pairs = data["pairs"]                    # (n_pairs, 2)
gr_all = data["gr"]                      # (n_frames, n_pairs, n_bins)
frame_indices = data["frame_indices"]

print("Shape:", gr_all.shape)            # (1000, 6, 600)

# calculate average RDF over frames
gr_mean = np.mean(gr_all, axis=0)        # (n_pairs, n_bins)

# plot
pair_names = ["{}–{}".format(a, b) for a, b in pairs]

plt.figure(figsize=(8, 3))
for i, name in enumerate(pair_names):
    plt.plot(r, gr_mean[i], label=name, linewidth=4)

plt.xlim(1.6, 7.0)
plt.xlabel("r (Å)")
plt.ylabel("g(r)")
plt.legend(frameon=False, fontsize=14, bbox_to_anchor=(0.85, 1))
# plt.title("Averaged RDF over {} frames".format(len(frame_indices)))
plt.tight_layout()
plt.savefig("rdf_average_300K_homo.png", dpi=300, transparent=True)
