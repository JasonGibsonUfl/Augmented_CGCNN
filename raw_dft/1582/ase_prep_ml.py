from ase.io import read
import os

#atoms.write('2.cif', format='vasp')
s_list = []

outcars = ['OUTCAR_1','OUTCAR_2','OUTCAR_3','OUTCAR_4','OUTCAR']
for outcar in outcars:
    if os.path.isfile(outcar):
        s_list = s_list + read(outcar,index=':',format='vasp-out')

mat_id = 1582
file_name = f'mldir/{mat_id}'
for i,struc in enumerate(s_list):
    struc.write(f'{file_name}_{i}.poscar',format = 'vasp')
    with open(f'{file_name}_{i}.energy', "w") as f:
        f.write(str(struc.get_total_energy()))
print(len(s_list))
