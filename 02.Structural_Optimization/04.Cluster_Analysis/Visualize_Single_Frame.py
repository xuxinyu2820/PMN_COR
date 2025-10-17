import ase, ase.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from utilities import *


latt_size = [12, 12, 12]
l1, l2, l3 = latt_size
traj_folder = '../03.12X12X12/600K/trajs'

## load the file
frame_idx = 4003  
file_path = os.path.join(traj_folder, f'f{frame_idx}.lmp')
atoms = ase.io.read(file_path, format='lammps-data', atom_style='atomic')

## get the effective lattice of B-site atoms
syms_lmp = atoms.get_chemical_symbols()
syms = syms_lmp.copy()
for idx, s in enumerate(syms_lmp):
    if s == 'H':
        syms[idx] = 'Mg'
    elif s == 'Be':
        syms[idx] = 'Pb'
    elif s == 'He':
        syms[idx] = 'Nb'
    elif s == 'Li':
        syms[idx] = 'O'
    else:
        raise ValueError('Invalid symbol')
atoms.set_chemical_symbols(syms)
lattice_idx = get_effective_lattice(atoms, latt_size, ['Mg', 'Nb'])
print(lattice_idx.shape)

## initialize the lattice values (0 means Mg, 1 means Nb)
lattice_values = np.zeros_like(lattice_idx, dtype=int)
for i in range(l1):
    for j in range(l2):
        for k in range(l3):
            Bsite_sym = syms[lattice_idx[i,j,k]]
            if Bsite_sym == 'Mg':
                lattice_values[i,j,k] = 0
            elif Bsite_sym == 'Nb':
                lattice_values[i,j,k] = 1
            else:
                raise ValueError('Invalid B-site symbol, check lattice site assignment')

## find those Nb sites that are embedded in Nb clusters (neighboring at most one Mg), and those Nb sites that are in 1:1 chemically ordered regions (neighboring at least five Mg)
## also find those Mg sites that have neighboring Mg atoms
CRO_flag = np.zeros_like(lattice_values, dtype=int)
cluster_flag = np.zeros_like(lattice_values, dtype=int)
Mgpair_flag = np.zeros_like(lattice_values, dtype=int)
for i in range(l1):
    for j in range(l2):
        for k in range(l3):
            if lattice_values[i,j,k] == 0:
                neighbor_values = find_neighbor_values(lattice_values, i, j, k)
                if neighbor_values.count(0) > 0:
                    Mgpair_flag[i,j,k] = 1
            else:
                neighbor_values = find_neighbor_values(lattice_values, i, j, k)
                if neighbor_values.count(0) <= 1:
                    cluster_flag[i,j,k] = 1
                if neighbor_values.count(0) >= 5:
                    CRO_flag[i,j,k] = 1
ncells = l1 * l2 * l3
n_Nb_atoms = lattice_values.sum()
n_Nb_atoms_in_cluster = cluster_flag.sum()
n_Nb_atoms_in_CRO = CRO_flag.sum()
n_Mg_atoms_in_Mgpair = Mgpair_flag.sum()
print(f'Number of Nb atoms: {n_Nb_atoms}', f'number of total perovskite cells: {ncells}')
print(f'Number of Nb atoms in cluster: {n_Nb_atoms_in_cluster}', f'fraction of Nb atoms in cluster: {n_Nb_atoms_in_cluster / n_Nb_atoms}')
print(f'Number of Nb atoms in CRO: {n_Nb_atoms_in_CRO}', f'fraction of Nb atoms in CRO: {n_Nb_atoms_in_CRO / n_Nb_atoms}')
print(f'Number of Mg atoms in Mgpair: {n_Mg_atoms_in_Mgpair}', f'fraction of Mg atoms in Mgpair: {n_Mg_atoms_in_Mgpair / n_Nb_atoms}')
 
## visualize a few cross sections of cluster_flag
## choose a colormap that is white for 0, blue for 1
cmap = mpl.colormaps['Blues']
fig, axs = plt.subplots(1, l1, figsize=(5*l3, 5))
for k in range(l3):
    axs[k].imshow(cluster_flag[:,:,k], cmap=cmap)
    nb_flag = lattice_values[:,:,k]
    axs[k].scatter(np.where(nb_flag)[1], np.where(nb_flag)[0], color='red', s=10)
    axs[k].set_title(f'Cross section, Layer {k}')


## make a centered head title
fig.suptitle(f'Red dots: Nb atoms, Blue blocks: Nb-cluster')

plt.savefig(f'./Figure_Cross_Sections/cluster_cross_sections_f{frame_idx}_xy.png')
plt.close()


## analyze the distributino of cluster size
visited = np.zeros_like(cluster_flag, dtype=bool)
all_clusters = []

for i in range(l1):
    for j in range(l2):
        for k in range(l3):
            if cluster_flag[i][j][k] == 1 and visited[i][j][k] == False:
                # New cluster found, start BFS
                current_cluster = []
                queue = [(i, j, k)]
                visited[i][j][k] = True

                while len(queue) > 0:
                    ci, cj, ck = queue.pop(0)
                    current_cluster.append((ci, cj, ck))

                    # Define neighbors again for traversal
                    neighbors = get_neighbor_indices(ci, cj, ck, l1, l2, l3)

                    for ni, nj, nk in neighbors:
                        if cluster_flag[ni][nj][nk] == 1 and visited[ni][nj][nk] == False:
                            visited[ni][nj][nk] = True
                            queue.append((ni, nj, nk))
                
                all_clusters.append(current_cluster)

cluster_size_list = []
for cluster in all_clusters:
    cluster_size_list.append(len(cluster))
print(f'Number of clusters: {len(all_clusters)}')
print(cluster_size_list)
print(f'Number of Nb atoms in clusters: {sum(cluster_size_list)}')

## plot the distribution of cluster size
plt.hist(cluster_size_list, bins=np.arange(1, max(cluster_size_list) + 1))
plt.xlabel('Cluster Size')
plt.ylabel('Frequency')
plt.title('Distribution of Cluster Size')
plt.savefig(f'Figure_Size_Distribution/cluster_size_distribution_f{frame_idx}.png')
plt.close()

## plot voxel plot of the largest cluster
largest_cluster = all_clusters[np.argmax(cluster_size_list)]
cluster_array = np.zeros((l1, l2, l3), dtype=bool)
for i, j, k in largest_cluster:
    cluster_array[i,j,k] = True
## plot the voxel plot
ax = plt.figure().add_subplot(projection='3d')
colors = np.empty(cluster_array.shape, dtype=object)
colors[cluster_array] = 'red'
colors[~cluster_array] = 'white'
# ax.voxels(cluster_array, facecolors=colors, edgecolor='k', alpha=0.5)
ax.voxels(cluster_array, facecolors=(0.5, 0.7, 1.0, 0.4), edgecolor=None, linewidth=0)
ax.grid(False)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_edgecolor((1, 1, 1, 0))
    axis.pane.set_facecolor((1, 1, 1, 0))

ax.set_proj_type('ortho')
ax.set_title(f'Largest Cluster')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_axis_off()
ax.view_init(elev=30, azim=0)
plt.savefig(f'Figure_Largest_Cluster/largest_cluster_voxel_plot_f{frame_idx}_elev30_azim0.png')

## plot a few more by changing camera angle
ax.view_init(elev=30, azim=45)
plt.savefig(f'Figure_Largest_Cluster/largest_cluster_voxel_plot_f{frame_idx}_elev30_azim45.png')
ax.view_init(elev=30, azim=135)
plt.savefig(f'Figure_Largest_Cluster/largest_cluster_voxel_plot_f{frame_idx}_elev30_azim135.png')
ax.view_init(elev=30, azim=225)
plt.savefig(f'Figure_Largest_Cluster/largest_cluster_voxel_plot_f{frame_idx}_elev30_azim225.png')

# ## save the positions of the cluster atoms
# z_layer = 9  # 固定 z 层
# cluster_layer = cluster_flag[:, :, z_layer]  # shape (l1, l2)，x-y 切片

# plt.figure(figsize=(6,6))

# # 自定义 colormap：0=透明，1=浅蓝色
# cmap = mpl.colors.ListedColormap([
#     (0, 0, 0, 0),               # 0 → 完全透明
#     (0.65, 0.78, 0.91, 0.60)    # 1 → 浅蓝，透明度适中
# ])

# # 用 imshow 画二维 mask
# plt.imshow(
#     cluster_layer.T,        # 注意转置：使 x 对应横轴，y 对应纵轴
#     origin='lower',         # 让 (0,0) 在左下角，符合笛卡尔坐标直觉
#     cmap=cmap,
#     interpolation='none'    # 不模糊，保持方块像素感
# )

# plt.axis('equal')
# plt.axis('off')
# plt.tight_layout()

# # 保存透明背景图
# plt.savefig(f"nb_cluster_z{z_layer}_blocks.png", dpi=300, transparent=True)
# plt.close()