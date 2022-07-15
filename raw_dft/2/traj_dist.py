from ase.io import read
import os
from pymatgen.io.vasp import Vasprun
from shutil import copy2
from tqdm import tqdm
import warnings
import numpy as np
from ase.data import covalent_radii
warnings.filterwarnings("ignore")

def write_traj(write_dir='mldir'):
    """Writes every structure and energy produced in a relaction trajectory in order.

    Parameters
    ----------
    mat_id: int
        numeric directory name of where the relaxation exists
    write_dir: str
        directoy to right structures to    
    """
    s_list = []
    outcars = ['OUTCAR_1','OUTCAR_2','OUTCAR_3','OUTCAR_4','OUTCAR']
    #outcars = ['OUTCAR']
    #copy2(f'{mat_id}/POSCAR','.')
    for outcar in outcars:
        outcar_name = f'./{outcar}'
        #outcar_name = f'{outcar}'
        if os.path.isfile(outcar_name):
            s_list = s_list + read(outcar_name,index=':',format='vasp-out')
            break
    return s_list
    #os.chdir('..')
    #file_name = f'{write_dir}/{mat_id}'
    #for i,struc in enumerate(s_list):
        #struc.write(f'{file_name}_{i}.poscar',format = 'vasp')
        #with open(f'{file_name}_{i}.energy', "w") as f:
           #f.write(str(struc.get_total_energy()))

s_list = write_traj()
start = s_list[0]#.get_positions()
end = s_list[-1]#.get_positions()
diff = start.get_positions()-end.get_positions()
diff_norm = np.linalg.norm(diff,axis=1)
print(diff_norm)
for dn,atom in zip(diff_norm,start):
    print(f'dn = {dn:.3f}\t rad = {covalent_radii[atom.number]}\tel = {atom.symbol}')

