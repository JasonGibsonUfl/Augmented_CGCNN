import os
from distutils.dir_util import copy_tree
from tqdm import tqdm
dirs = next(os.walk('.'))[1] # This is a list of sub-directories in the current directory.
dirs_i = sorted([int(d) for d in dirs if d.isdigit()]) # This removes any directories that are not numbers and sorts the list
for d in tqdm(dirs_i):
    try:
        os.remove(f'{str(d)}/PROCAR')
    except:
        pass
    try:
        os.remove(f'{str(d)}/MP_inputs/PROCAR')
    except:
        pass
    name = 'CHG'
    try:
        os.remove(f'{str(d)}/{name}')
    except:
        pass
    try:
        os.remove(f'{str(d)}/MP_inputs/{name}')
    except:
        pass
    name = 'CHGCAR'
    try:
        os.remove(f'{str(d)}/{name}')
    except:
        pass
    try:
        os.remove(f'{str(d)}/MP_inputs/{name}')
    except:
        pass
    name = 'vasprun.xml'
    try:
        os.remove(f'{str(d)}/{name}')
    except:
        pass
    try:
        os.remove(f'{str(d)}/MP_inputs/{name}')
    except:
        pass

print(dirs_i)
