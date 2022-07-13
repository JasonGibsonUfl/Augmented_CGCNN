"""
Augments the entire materialsproject database
"""
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


def get_gmm():
    """Fits a Gaussian mixture model to the distribution shown in Figure 2 of arXiv:2202.13947
    Returns
    -------
    gm: sklearn.mixture.GaussianMixture
      Gaussian mixture model that has been fit to the distribution shown in Figure 2 of arXiv:2202.13947
    """
    dist = pkl.load(open("dists.pkl", "rb"))
    return GaussianMixture(n_components=20, random_state=19).fit(
        np.array(dist).reshape(-1, 1)
    )

def perturb(struct, data):
    """Perturbs the atomic cordinates of a structure
    Parameters
    ----------
    struct: pymatgen.core.Structure
      pymatgen structure to be perturbed
    data: np.ndarray
      numpy array of possible magnitudes of perturbation
    Returns
    -------
    struct_per: pymatgen.core.Structure
      Perturbed structure
    """

    def get_rand_vec(dist):
        """Returns vector used to pertrub structure
        Parameters
        ----------
        dist: float
          Magnitude of perturbation
        Returns
        -------
        vector: np.ndarray
          Vector whos direction was randomly sampled from random sphere and magnitude is defined by dist
        """
        vector = np.random.randn(3)
        vnorm = np.linalg.norm(vector)
        return vector / vnorm * dist if vnorm != 0 else get_rand_vec(dist)

    struct_per = struct.copy()
    for i in range(len(struct_per._sites)):
        ind = np.random.randint(0, 10 ** 8)
        dist = data[ind][0]
        struct_per.translate_sites([i], get_rand_vec(dist), frac_coords=False)
    return struct_per


def write_structs(results, dir_name="train_data"):
    """Write original and perturbed structure to a directory
    Parameters
    ----------
    results: List or np.ndarray
      query results from materials project
    dir_name: String
      Name of directory to write cif files to
    Returns
    -------
    E: List
      list in format to write id_prop.csv file
    """
    ind = 0
    E = []
    for entry in results:
        ind += 1
        struct = entry["final_structure"]
        struct.to(fmt="cif", filename=f"{dir_name}/{ind}.cif")
        E.append([ind, entry["formation_energy_per_atom"]])
        ind += 1
        struct_per = perturb(struct, data)
        struct_per.to(fmt="cif", filename=f"{dir_name}/{ind}.cif")
        E.append([ind, entry["formation_energy_per_atom"]])
    return E


def write_dir(E, dir_name="train_data"):
    """Writes id_prop.csv file for training of CGCNN
    Parameters
    ----------
    E: List
      list with the first dimension coresponding to the structures index and the
      second corresponding to the structures formation energy per atom
    dir_name: String
      Directory to write id_prop.csv to.
    """
    with open(f"{dir_name}/id_prop.csv", "w", newline="") as file:
        wr = csv.writer(file)
        wr.writerows(E)


if __name__ == "__main__":
    MP_API_KEY = None
    gm = get_gmm()
    data = gm.sample(10 ** 8)[0]
    properties = ["final_structure", "formation_energy_per_atom"]
    criteria = {"formation_energy_per_atom": {"$exists": True}}
    #criteria = {"formation_energy_per_atom": {"$lt": -3.6}}

    with MPRester(MP_API_KEY) as mpr:
        results = mpr.query(criteria, properties)
    np.random.shuffle(results)

    train, test = np.split(results, [int(0.8 * len(results))])
    E = write_structs(train)
    write_dir(E)

    E = write_structs(test, "validation_data")
    write_dir(E, "validation_data")
