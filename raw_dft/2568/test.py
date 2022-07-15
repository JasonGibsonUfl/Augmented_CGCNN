from pymatgen.analysis.bond_valence import *
from pymatgen.core import Structure

s = Structure.from_file('CONTCAR')
c = s.composition
print(c)
print(get_z_ordered_elmap(c))
print(c.oxi_state_guesses())
print(c.get_el_amt_dict())
print(c.hill_formula)

s.add_oxidation_state_by_guess()
spec = s.species
for sp in spec:
    print(sp.element)
    print(sp.oxi_state)
