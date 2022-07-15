import os
from distutils.dir_util import copy_tree
from tqdm import tqdm
dirs = next(os.walk('.'))[1] # This is a list of sub-directories in the current directory.
dirs_i = sorted([int(d) for d in dirs if d.isdigit()]) # This removes any directories that are not numbers and sorts the list
for d in tqdm(dirs_i):
    for i in range(6):
        try:
            os.remove(f'{str(d)}/OUTCAR')
        except:
            pass
        try:
            os.remove(f'{str(d)}/MP_inputs/OUTCAR')
        except:
            pass
print(dirs_i)
