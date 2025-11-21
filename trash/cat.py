import numpy as np

x1 = np.loadtxt('/global/homes/p/pinchenx/cfs_m5025/Ferroic/PMN/Finite_Temp_MC/12X12X12/logs_12_1/swap_energies.csv', skiprows=1, delimiter=',')
x2 = np.loadtxt('/global/homes/p/pinchenx/cfs_m5025/Ferroic/PMN/Finite_Temp_MC/12X12X12/logs_12_2/swap_energies.csv', skiprows=1, delimiter=',')
x3 = np.loadtxt('/global/homes/p/pinchenx/cfs_m5025/Ferroic/PMN/Finite_Temp_MC/12X12X12/logs_12_3/swap_energies.csv', skiprows=1, delimiter=',')

x = np.concatenate([x1, x2, x3], axis=0)
print(x.shape)
np.savetxt('energy_of_all_trials.csv', x, delimiter=',', header='last_energy, proposed_energy', fmt='%.10f')


y1 = np.loadtxt('/global/homes/p/pinchenx/cfs_m5025/Ferroic/PMN/Finite_Temp_MC/12X12X12/logs_12_1/energy_vs_iter.csv', skiprows=1, delimiter=',')
y2 = np.loadtxt('/global/homes/p/pinchenx/cfs_m5025/Ferroic/PMN/Finite_Temp_MC/12X12X12/logs_12_2/energy_vs_iter.csv', skiprows=2, delimiter=',')
y3 = np.loadtxt('/global/homes/p/pinchenx/cfs_m5025/Ferroic/PMN/Finite_Temp_MC/12X12X12/logs_12_3/energy_vs_iter.csv', skiprows=2, delimiter=',')
y2[:,0] += 100000
y3[:,0] += 200000
y2[:,1] += y1[-1,1]
y3[:,1] += y2[-1,1]
y = np.concatenate([y1, y2, y3], axis=0)
print(y.shape)
np.savetxt('energy_accepted.csv', y, delimiter=',', header='iter,delta_E_meV_per_atom', fmt='%d,%.10f')