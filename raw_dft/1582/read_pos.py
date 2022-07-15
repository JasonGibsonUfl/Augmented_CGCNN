with open('POSCAR') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    
ele = lines[5].split()
num = lines[6].split()
species = [f'{e}{n}' for e, n in zip(ele,num)]
print({'species': species})
#species = ['Sr1', 'Nb1', 'H11']
#ion = [6, 1, 5, 1]

