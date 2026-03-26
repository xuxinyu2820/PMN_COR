import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

# plot settings
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20

temps = [100,250,400,700]

base_dir = "/global/homes/x/xinyuxu/m5025/Ferroic/PMN/histogram"
fname_tpl = "Asite_displacement_AO12_{}K.csv"

lim = 0.8
bins = 100

# store histograms
histograms = {}
edges = {}

global_max = 0

# ---------- PASS 1 : compute histograms ----------
for T in temps:

    csv_path = f"{base_dir}/{fname_tpl.format(T)}"

    pts = []

    with open(csv_path,"r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            x = float(row["dx"])
            y = float(row["dy"])
            pts.append((x,y))

    pts = np.array(pts)

    h,xedges,yedges = np.histogram2d(
        pts[:,0],pts[:,1],
        bins=bins,
        range=[[-lim,lim],[-lim,lim]],
        density=True
    )

    histograms[T] = h
    edges[T] = (xedges,yedges)

    global_max = max(global_max, h.max())

print("Global histogram max =",global_max)

# ---------- PASS 2 : plotting ----------
fig,axes = plt.subplots(
    1,4,
    figsize=(16,4),
    sharex=True,
    sharey=True,
    gridspec_kw={"wspace":0.15}
)

cmap="gnuplot2_r"

for j,T in enumerate(temps):

    ax = axes[j]

    h = histograms[T]
    xedges,yedges = edges[T]

    # global normalization
    h_norm = h / global_max

    im = ax.imshow(
        h_norm.T,
        origin="lower",
        extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="equal"
    )

    ax.set_title(f"{T} K")
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)

for ax in axes:
    ax.set_xticks([-0.5,0,0.5])
    ax.set_xlabel(r"$d^x_{Pb}$ (Å)")

axes[0].set_ylabel(r"$d^y_{Pb}$ (Å)")

# colorbar
cbar = fig.colorbar(
    im,
    ax=axes.ravel().tolist(),
    location="right",
    fraction=0.046,
    pad=0.02
)

cbar.set_ticks([0,0.5,1])
cbar.set_ticklabels(["0","0.5","1"])

plt.savefig("Asite_displacement.png",dpi=300,bbox_inches="tight")

print("saved to Asite_displacement.png")