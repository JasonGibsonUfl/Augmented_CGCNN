from pymatgen.ext.matproj import MPRester
import csv
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import random
import pickle as pkl
from sklearn.mixture import GaussianMixture

np.random.seed(19)
random.seed(19)

dist = pkl.load(open('dists.pkl','rb'))
dists = []
for d in dist:
    for v in d:
        dists.append(v)

gm = GaussianMixture(n_components=20, random_state=19).fit(np.array(dists).reshape(-1, 1))
data = gm.sample(10**8)[0]

def perturb(struct):
    def get_rand_vec(dist):
        # deals with zero vectors.
        vector = np.random.randn(3)
        vnorm = np.linalg.norm(vector)
        return vector / vnorm * dist if vnorm != 0 else get_rand_vec(dist)

    struct_per = struct.copy()
    for i in range(len(struct_per._sites)):
        ind = np.random.randint(0,10**8)
        dist = data[ind][0]
        struct_per.translate_sites([i], get_rand_vec(dist), frac_coords=False)
    return struct_per

properties = ['final_structure','formation_energy_per_atom']
criteria = {'formation_energy_per_atom': {'$exists': True}}

cs = 500
with MPRester() as mpr:
    results = mpr.query(criteria, properties)

np.random.shuffle(results)

def filtered(ent):
    return ent

n_cores = 32

parsed_filter = list(tqdm(mp.Pool(n_cores).imap(filtered, results),total = len(results)))
print('DONE PARSE')
results2 = []
j = 0
for i in parsed_filter:
    if type(i) ==type({}):
        i['ind'] = j
        j+=1
        results2.append(i)

def per_all(ent):
    E = []
    struct = ent['final_structure']
    ids = ent['ind']*11+1
    struct.to(fmt='cif',filename=f'train/{ids}.cif')
    E.append([ids,ent['formation_energy_per_atom']])
    ids +=1
    for j in range(10):
        struct_per = perturb(struct)
        struct_per.to(fmt='cif',filename=f'train/{ids}.cif')
        E.append([ids,ent['formation_energy_per_atom']])
        ids +=1
    return E

def per_all_t(ent):
    E = []
    struct = ent['final_structure']
    ids = ent['ind']*11+1
    struct.to(fmt='cif',filename=f'test/{ids}.cif')
    E.append([ids,ent['formation_energy_per_atom']])
    ids +=1
    for j in range(10):
        struct_per = perturb(struct)
        struct_per.to(fmt='cif',filename=f'test/{ids}.cif')
        E.append([ids,ent['formation_energy_per_atom']])
        ids +=1
    return E

train, test = np.split(results2, [int(.8*len(results2))])

train_l = list(tqdm(mp.Pool(n_cores).imap(per_all,train),total = len(train)))

E = []
for i in train_l:
    print(i)
    for j in i:
        E.append(j)

with open('train/id_prop.csv', 'w', newline='') as file:
    wr = csv.writer(file)
    wr.writerows(E)

test_l = list(tqdm(mp.Pool(n_cores).imap(per_all_t,test),total = len(test)))

E = []
for i in test_l:
    print(i)
    for j in i:
        E.append(j)

with open('test/id_prop.csv', 'w', newline='') as file:
    wr = csv.writer(file)
    wr.writerows(E)

