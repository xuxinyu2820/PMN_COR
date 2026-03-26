import os
import sys
import warnings
import numpy as np
import ase
import ase.io
from ase.geometry.analysis import Analysis
from ase.neighborlist import NeighborList, natural_cutoffs, NewPrimitiveNeighborList  ## use NewPrimitiveNeighborList <- linear scaling
from ase.geometry import get_distances



class PMN_perovskite:
    def __init__(self, ABCO3=['Pb','Mg','Nb','O'], born_charges=[3.7140, 5.4879, -3.3551, -2.9234]):
        if len(ABCO3) == 4:
            self.type_A, self.type_B, self.type_C, self.type_O = ABCO3
        else:
            raise ValueError("Please specify the atom types of perovskites with the conventional order, e.g. ['Pb','Mg','Nb','O'] for lead magnesium niobate")
        self.born_charges = born_charges ##   order: [A,B,Ol,Ot]
        # if len(born_charges) == 3:
        #     self.chg_A, self.chg_B, self.chg_O = born_charges
        #     if self.chg_A + self.chg_B + 3*self.chg_O != 0:
        #         print('the total born_charges does not balance. This system is globally charged')
        # else:
        #     raise ValueError("Please specify the ionic charge of A,B,O with the same order as the 'ABO3' argument")
  
    def _create_disordered_domain_small(self, supercell=[3,3,3], a=4.0 ):
        '''
        Building disordered PMN configurations perserving the stoichiometry. Can be used as the building block of a larger supercell where stoichiometry is locally preserved.
        The generated configuration contains a random distribution of B2+ and B5+ ions in a 2:1 ratio 
        Args:
            supercell: the size of the supercell. Should be small.
            a: lattice constant in Angstrom
        '''
        conf = ase.Atoms( 
            symbols= self.type_A+self.type_B+3*self.type_O,
            scaled_positions=[ 
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5]],
            cell=[a, a, a],
            pbc=True)
        conf = conf.repeat(supercell)
        ncell = supercell[0]*supercell[1]*supercell[2]
        syms = conf.get_chemical_symbols()
        Bsites = [idx for idx, s in enumerate(syms) if s==self.type_B]
        ndopants = ncell*2//3
        if (ndopants*3-ncell*2) != 0:
            raise NotImplementedError('the given supercell should be charge neutral')
        replaced = np.random.choice(Bsites, ndopants, replace=False)
        for idx in replaced:
            syms[idx] = self.type_C
        conf.set_chemical_symbols(syms)
        return conf
     
    def _create_ordered_domain_small(self, supercell=[3,2,1], a=4.0 ):
        '''
        Building partially-ordered PMN configurations perserving the stoichiometry. Can be used as the building block of a larger supercell where stoichiometry is locally preserved.:
        One of the B-sublattices is occupied exclusively by B5+ ions, 
        while the other one contains a random distribution of B2+ and B5+ ions in a 2:1 ratio 
        so that the local stoichiometry is preserved
        '''
        ss = supercell
        ncell = ss[0]*ss[1]*ss[2]
        if ncell % 3 != 0:
            raise ValueError('number of cell should be divided by 3 to guarantee charge neutrality')
        pos_cube = np.array([ 
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5]])
        sym_Bcube = [self.type_A, self.type_B, self.type_O, self.type_O, self.type_O,]
        sym_Ccube = [self.type_A, self.type_C, self.type_O, self.type_O, self.type_O,]
        ## sqrt(2) X sqrt(2) X 2   building block.   B-atom and C-atom are put on bipartite sublattices.
        block_cell = np.array([
                [ 1, 1, 0 ],
                [-1, 1, 0 ],
                [ 0, 0, 2 ],
                ])
        block = ase.Atoms( 
            symbols= sym_Bcube + sym_Ccube + sym_Ccube + sym_Bcube,
            positions=np.concatenate([
                pos_cube, 
                pos_cube+np.array([1,0,0]),
                pos_cube+np.array([0,0,1]),
                pos_cube+np.array([1,0,1]),
                ]),
            cell = block_cell,
            pbc=True)
        ## rotate and replicate 
        block.rotate(-45,'z',rotate_cell=True)
        block.set_cell(block.get_cell()*a, scale_atoms=True)
        conf = block.repeat(supercell)

        ## substitution of the disordered sublattice.
        ncell = supercell[0]*supercell[1]*supercell[2]
        syms = conf.get_chemical_symbols()
        Bsites = [idx for idx, s in enumerate(syms) if s==self.type_B]
        ndopants = len(Bsites)//3
        if  ndopants*3 != len(Bsites):
            raise NotImplementedError('the given supercell should be charge neutral')
        replaced = np.random.choice(Bsites, ndopants, replace=False)
        for idx in replaced:
            syms[idx] = self.type_C
        conf.set_chemical_symbols(syms)
        return conf
     
    def create_disordered_domain(self, supercell=[3,3,3], a=4.0 ):
        '''
        Building large disordered PMN configurations perserving the stoichiometry locally. 
        '''
        ss = supercell
        ncell = ss[0]*ss[1]*ss[2]
        if ncell % 3 != 0:
            raise ValueError('number of cell should be divided by 3 to guarantee charge neutrality')
        if ncell <= 27:
            return self._create_disordered_domain_small(ss, a)
        else:
            if (ss[0] % 3 == 0) and (ss[1] % 3 == 0) and (ss[2] % 3 == 0):
                nx = ss[0]//3
                ny = ss[1]//3
                nz = ss[2]//3
                nblocks = nx*ny*nz
                syms = []
                pos = []
                for ix in range(nx):
                    for iy in range(ny):
                        for iz in range(nz):
                            block = self._create_disordered_domain_small( supercell=[3,3,3], a=a )
                            syms += block.get_chemical_symbols()
                            new_pos = block.get_positions() + np.array([ix*3*a, iy*3*a, iz*3*a,])
                            pos.append(new_pos)
                conf = ase.Atoms( 
                    symbols= syms,
                    positions=np.concatenate(pos),
                    cell=np.array(ss)*a,
                    pbc=True)
                return conf
            else:
                raise NotImplementedError('currently supports only supercell stacked by 3X3X3 building blocks to maintain charge neutrality')

    def create_ordered_domain(self, supercell=[3,2,1], a=4.0 ):
        '''
        Building large partially-ordered PMN configurations perserving the stoichiometry locally. 
            One of the B-sublattices is occupied exclusively by B5+ ions, 
            while the other one contains a random distribution of B2+ and B5+ ions in a 2:1 ratio 
        '''
        ss = supercell
        ncell = ss[0]*ss[1]*ss[2]
        if ncell % 3 != 0:
            raise ValueError('number of cell should be divided by 3 to guarantee charge neutrality')
        if ncell <= 18:
            return self._create_ordered_domain_small(ss, a)
        else:
            if (ss[0] % 3 == 0) and (ss[1] % 3 == 0) and (ss[2] % 2 == 0):
                nx = ss[0]//3
                ny = ss[1]//3
                nz = ss[2]//2
                nblocks = nx*ny*nz
                syms = []
                pos = []
                for ix in range(nx):
                    for iy in range(ny):
                        for iz in range(nz):
                            block = self._create_ordered_domain_small( supercell=[3,3,2], a=a )
                            syms += block.get_chemical_symbols()
                            cell = block.get_cell() 
                            new_pos = block.get_positions() +  ix*cell[0] + iy*cell[1] + iz*cell[2] 
                            pos.append(new_pos)
                conf = ase.Atoms( 
                    symbols= syms,
                    positions=np.concatenate(pos),
                    cell=np.array([nx*cell[0], ny*cell[1], nz*cell[2]]),
                    pbc=True)
                return conf
            else:
                raise NotImplementedError('currently supports only supercell stacked by 3sqrt(2)X3sqrt(2)X4 building blocks to maintain charge neutrality')


    def _create_conditioned_domain_small(self,supercell=[12, 12, 12],a=4.05,condition=None,fraction=2/3,seed=None):
        nx, ny, nz = supercell
        conf = ase.Atoms(
            symbols=self.type_A + self.type_C + 3*self.type_O,
            scaled_positions=[
                [0.0, 0.0, 0.0],   # A
                [0.5, 0.5, 0.5],   # C
                [0.5, 0.5, 0.0],   # O1
                [0.0, 0.5, 0.5],   # O2
                [0.5, 0.0, 0.5],   # O3
            ],
            cell=[a, a, a],
            pbc=True,
        )

        conf = conf.repeat(supercell)

        if condition is None:
            def condition(i, j, k):
                is_A = ((j + k) % 2 == 0)
                return is_A if (i % 2) == 0 else (not is_A)

        syms = conf.get_chemical_symbols()
        frac = conf.get_scaled_positions(wrap=True)

        Bsites = [idx for idx, s in enumerate(syms) if s == self.type_C]

        eps = 1e-12
        candidates = []
        for idx in Bsites:
            fx, fy, fz = frac[idx]
            i = int(np.floor(fx * nx + eps))
            j = int(np.floor(fy * ny + eps))
            k = int(np.floor(fz * nz + eps))
            if condition(i, j, k):
                candidates.append(idx)

        rng = np.random.default_rng(seed)
        n_pick = int(round(len(candidates) * fraction))
        n_pick = min(n_pick, len(candidates))
        picked = set(rng.choice(candidates, size=n_pick, replace=False)) if n_pick > 0 else set()

        for idx in candidates:
            syms[idx] = self.type_B if idx in picked else self.type_C

        conf.set_chemical_symbols(syms)

        return conf


    def _create_Nb_domain(self, supercell=[3,3,3], a=4.0):
        """
        Build an ABO3 supercell where ALL B-site atoms are Nb (self.type_C).
        Note: This does NOT preserve PMN's 1:2 stoichiometry; it's a pure-Nb B-sublattice.
        """
        conf = ase.Atoms(
            symbols=self.type_A + self.type_C + 3*self.type_O,  # B位直接用Nb
            scaled_positions=[
                [0.0, 0.0, 0.0],   # A
                [0.5, 0.5, 0.5],   # B (Nb)
                [0.5, 0.5, 0.0],   # O
                [0.0, 0.5, 0.5],   # O
                [0.5, 0.0, 0.5],   # O
            ],
            cell=[a, a, a],
            pbc=True,
        )
        conf = conf.repeat(supercell)
        return conf
 

if __name__ == "__main__":
    pmn_factory  = PMN_perovskite(['Pb','Mg','Nb','O'])
    type_map = ['Mg','Nb', 'O', 'Pb']  #  alphabatic order

    # large_ordered_no_local_stoich = pmn_factory._create_ordered_domain_small(supercell=[12,12,8], a=4.05)
    # ase.io.write('large_ordered_no_local_stoich.lmp',large_ordered_no_local_stoich,format='lammps-data', specorder=type_map)

    supercell = [6,6,6]
    # disordered = pmn_factory.create_disordered_domain( supercell=[10,10,6], a=4.05 )
    disordered = pmn_factory._create_disordered_domain_small( supercell=[6,6,6], a=4.05 )

    ase.io.write('disordered_L{}X{}X{}.lmp'.format(supercell[0], supercell[1], supercell[2]), disordered, format='lammps-data', specorder=type_map)

 