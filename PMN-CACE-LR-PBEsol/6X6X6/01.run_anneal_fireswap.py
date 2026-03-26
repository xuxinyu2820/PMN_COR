import sys
import os
import pickle
import numpy as np
import torch
import copy
from time import time

import cace
from cace.calculators import CACECalculator
from cace.models.atomistic import NeuralNetworkPotential

from ase import units
from ase.io import read, write
from ase.constraints import ExpCellFilter
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter

############# 论文参数设置 #############
# 退火温度计划 (从高温到低温)
temp_schedule = [600]

# 每个温度运行的步数
steps_per_temp = 100000

#  力收敛精度
relax_fmax = 0.02

#  单次 FIRE 弛豫的最大允许步数 (保持较小以提高采样效率)
max_relax_steps = 200

# 文件路径
cuda_device = "cuda"
input_traj = './1080.xyz'   # 读取上一阶段轨迹
output_file = 'fireswap_anneal.xyz'
log_file = 'fireswap_anneal.log'

###################################

# 1. 载入 CACE 模型
print("正在载入 CACE 模型...")
cace_nnp_noStress = torch.load(
    '/home/svu/e1698324/scratch/PMNPT/cace_test_new_2/md/model/cace_model_10000/CACE_NNP_phase_4.pth',
    weights_only=False,
    map_location=torch.device('cuda')
)
with open('/home/svu/e1698324/scratch/PMNPT/cace_test_new_2/md/model/cace_model_10000/avge0.pkl', 'rb') as f:
    avge0 = pickle.load(f)

forces = cace.modules.Forces(
    energy_key='CACE_energy',
    forces_key='CACE_forces',
    stress_key='CACE_stress',
)

output_modules = cace_nnp_noStress.output_modules[0:4] + [forces]

cace_nnp = NeuralNetworkPotential(
    representation=cace_nnp_noStress.representation,
    output_modules=output_modules,
    keep_graph=True
)

for p1, p2 in zip(cace_nnp_noStress.parameters(), cace_nnp.parameters()):
    if p1.shape == p2.shape:
        p2.data = p1.data.clone()

cace_nnp = cace_nnp.to(torch.device(cuda_device))

calculator = CACECalculator(
    model_path=cace_nnp,
    device=cuda_device,
    energy_key='CACE_energy',
    forces_key='CACE_forces',
    stress_key='CACE_stress',
    compute_stress=True,
    atomic_energies=avge0,
)

# 2. 读取初始结构
print(f"读取上一阶段轨迹: {input_traj} (Last Frame)")
try:
    atoms = read(input_traj, index=-1)
except Exception as e:
    print(f"读取失败: {e}")
    sys.exit(1)

atoms.set_calculator(calculator)

# 3. 定义 FIRE 弛豫函数
def run_fire_relaxation(atoms_obj, fmax, steps=None):
    """
    执行 FIRE 几何优化。
    """
    ucf = FrechetCellFilter(atoms_obj)
    
    if steps is None:
        steps = 10000
    
    opt = FIRE(ucf, logfile=None, dt=0.1, maxstep=0.2)
    
    try:
        opt.run(fmax=fmax, steps=steps)
        return True
    except Exception as e:
        print(f"Relaxation Error: {e}")
        return False

def get_energy_float(atoms_obj):
    e = atoms_obj.get_potential_energy()
    if isinstance(e, np.ndarray):
        return float(e.item()) if e.size == 1 else float(e.sum())
    return float(e)

# 4. 初始化
symbols = atoms.get_chemical_symbols()
indices_mg = [i for i, s in enumerate(symbols) if s == 'Mg']
indices_nb = [i for i, s in enumerate(symbols) if s == 'Nb']

if os.path.exists(log_file): os.remove(log_file)
if os.path.exists(output_file): os.remove(output_file)

# 更新表头，增加 Temperature 列
with open(log_file, 'w') as f:
    f.write("GlobalStep,Temp(K),Energy(eV),Delta_E(eV),Accepted,Time(s)\n")

# --- 阶段 A: 初始结构的完全弛豫 ---
print("\n=== 阶段 A: 初始结构完全弛豫 ===")
run_fire_relaxation(atoms, fmax=relax_fmax, steps=None)
e_current = get_energy_float(atoms)
write(output_file, atoms, format='extxyz', append=True)
print(f"  初始能量: {e_current:.5f} eV")

# --- 阶段 B: 模拟退火循环 ---
print(f"\n=== 阶段 B: FIRE-Swap 模拟退火 ===")
print(f"   温度计划: {temp_schedule} K")
print(f"   每阶段步数: {steps_per_temp}")
print(f"   交换策略: 局域交换 (Cutoff=6.3A, NN+NNN)")

global_step = 0
swap_cutoff = 6.3  # 截断半径

# 外层循环：遍历温度
for current_temp in temp_schedule:
    kT = current_temp * units.kB
    print(f"\n>>> 切换温度至: {current_temp} K (kT = {kT:.4f} eV)")
    
    accept_count_stage = 0
    stage_start_time = time()
    
    # 内层循环：当前温度下的 MC 步数
    for i in range(1, steps_per_temp + 1):
        global_step += 1
        step_start = time()
        
        # 1. 备份
        old_positions = atoms.get_positions()
        old_cell = atoms.get_cell()
        
        # 2. 提出交换 (Propose Swap) - 局域模式
        idx_mg = np.random.choice(indices_mg)
        dists = atoms.get_distances(idx_mg, indices_nb, mic=True)
        valid_nb_indices = np.array(indices_nb)[dists < swap_cutoff]
        
        if len(valid_nb_indices) > 0:
            idx_nb = np.random.choice(valid_nb_indices)
        else:
            idx_nb = np.random.choice(indices_nb)
        
        current_symbols = np.array(atoms.get_chemical_symbols())
        current_symbols[idx_mg] = 'Nb'
        current_symbols[idx_nb] = 'Mg'
        atoms.set_chemical_symbols(current_symbols)
        
        # 3. 几何弛豫
        run_fire_relaxation(atoms, fmax=relax_fmax, steps=max_relax_steps)
        
        # 4. 计算能量差
        e_trial = get_energy_float(atoms)
        delta_e = e_trial - e_current
        
        # 5. Metropolis 判据
        if delta_e < 0:
            accepted = True
        else:
            p = np.exp(-delta_e / kT)
            accepted = (np.random.rand() < p)
        
        # 6. 更新或回滚
        if accepted:
            e_current = e_trial
            accept_count_stage += 1
            
            indices_mg.remove(idx_mg)
            indices_mg.append(idx_nb)
            indices_nb.remove(idx_nb)
            indices_nb.append(idx_mg)
            
            status = "ACC"
        else:
            atoms.set_cell(old_cell)
            atoms.set_positions(old_positions)
            
            current_symbols[idx_mg] = 'Mg'
            current_symbols[idx_nb] = 'Nb'
            atoms.set_chemical_symbols(current_symbols)
            
            status = "REJ"
    
        # 7. 记录日志 (每步或每隔几步)
        step_time = time() - step_start
        
        if global_step % 1 == 0:
            with open(log_file, 'a') as f:
                f.write(f"{global_step},{current_temp},{e_current:.5f},{delta_e:.5f},{accepted},{step_time:.2f}\n")
            
            # 为了避免刷屏，每10步打印一次控制台信息
            if global_step % 10 == 0:
                print(f"Step {global_step} (T={current_temp}K): {status}, dE={delta_e:.4f}, E={e_current:.4f}")
    
        # 每 50 步保存轨迹
        if global_step % 100 == 0:
            write(output_file, atoms, format='extxyz', append=True)

    # 每个温度阶段结束后的统计
    stage_duration = time() - stage_start_time
    print(f"--- 完成 T={current_temp}K. 接受率: {accept_count_stage/steps_per_temp:.2%}, 耗时: {stage_duration:.1f}s ---")

print("所有退火阶段完成。")
