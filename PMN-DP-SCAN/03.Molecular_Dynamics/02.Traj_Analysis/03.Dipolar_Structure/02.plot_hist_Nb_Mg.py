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

# data settings
temps = [100, 250, 400, 700]
base_dir = "/global/homes/x/xinyuxu/m5025/Ferroic/PMN/histogram"
fname_tpl = "displacement_classification_{}K.csv"

rows = [
    ("nb_5to6nb_xy", r"Nb (5–6 Nb), $d_x$–$d_y$", 9,  "gnuplot2_r"),
    ("nb_6mg_xy",    r"Nb (6 Mg), $d_x$–$d_y$",   16, "gnuplot2_r"),
    ("mg_nb6mg_xy",  r"Mg around Nb–6Mg, $d_x$–$d_y$", 25, "gnuplot2_r"),
]


data = {}  
for T in temps:
    csv_path = f"{base_dir}/{fname_tpl.format(T)}"
    per_file_buckets = {r[0]: [] for r in rows}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row["category"]
            if cat in per_file_buckets:
                x = float(row["comp1"])
                y = float(row["comp2"])
                per_file_buckets[cat].append((x, y))

    # save to main data dict
    for cat, pts in per_file_buckets.items():
        if len(pts) > 0:
            data[(cat, T)] = np.array(pts)
        else:
            data[(cat, T)] = np.empty((0, 2))

# plotting
fig, axes = plt.subplots(
    3, 4,
    figsize=(14, 10),
    sharex=True,
    sharey=True,
    gridspec_kw={
        "wspace": 0.12, 
        "hspace": 0.12,
    },
)

lim = 0.4
bins = 100

# colorbar
imgs_row = [None, None, None]

for i_row, (cat, row_title, vmax, cmap) in enumerate(rows):
    for j_col, T in enumerate(temps):
        ax = axes[i_row, j_col]
        pts = data[(cat, T)]

        if pts.shape[0] == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal", "box")
            continue

        # hist2d
        h, xedges, yedges = np.histogram2d(
            pts[:, 0], pts[:, 1],
            bins=bins,
            range=[[-lim, lim], [-lim, lim]],
            density=True
        )

        # normalize
        h_norm = h / float(vmax)
        h_norm[h_norm > 1.0] = 1.0

        # plot
        im = ax.imshow(
            h_norm.T,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap=cmap,
            vmin=0,
            vmax=1,
            aspect="equal",
        )

        imgs_row[i_row] = im

        if j_col == 0:
            ax.set_ylabel(r"$d^y_i$ (Å)")

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

for ax in axes.ravel():
    ax.set_xticks([-0.2, 0.0, 0.2])
    ax.set_xticklabels(['-0.2', '0.0', '0.2'])

# bottom row x label
for ax in axes[-1, :]:
    ax.set_xlabel(r"$d^x_i$ (Å)")

# colorbars
for i_row, im in enumerate(imgs_row):
    # all axes in this row
    row_axes = axes[i_row, :].ravel().tolist()
    cbar = fig.colorbar(
        im,
        ax=row_axes,
        location="right",
        fraction=0.046,
        pad=0.02
    )
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["0.0", "0.5", "1.0"])

plt.savefig("displacement_3x4_allT.png", dpi=300, bbox_inches="tight")
print("saved to displacement_3x4_allT.png")
