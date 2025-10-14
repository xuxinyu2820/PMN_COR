from fse.systems import PMN_perovskite as PMN
import ase
import ase.io
import os
pmn_factory  = PMN(['Pb','Mg','Nb','O'])
type_map = ['Mg','Nb', 'O', 'Pb']  #  alphabatic order
ndisordered = 64
nordered = 64

# for idx in range(ndisordered):
#     disordered = pmn_factory.create_disordered_domain( supercell=[3,3,3], a=4.0 )
#     ase.io.write('./disordered/POSCAR{:03d}'.format(idx),
#                 disordered,format='vasp',direct=False, sort=True,)


# for idx in range(nordered):
#     ordered = pmn_factory.create_ordered_domain( supercell=[3,3,1], a=4.0 )
#     ase.io.write('./ordered/POSCAR{:03d}'.format(idx),
#                 ordered,format='vasp',direct=False, sort=True,)

for idx in range(16):
    ordered = pmn_factory.create_ordered_domain( supercell=[3,3,2], a=4.0 )
    outdir = 'ordered332/{:03d}'.format(idx)
    if os.path.exists(outdir) is False:
        os.makedirs(outdir)
    ase.io.write(os.path.join(outdir,'POSCAR'),
                ordered,format='vasp',direct=False, sort=True,)
