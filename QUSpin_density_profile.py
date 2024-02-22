from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
import numpy as np
import matplotlib.pyplot as plt

import time
start_time = time.time()

#Parameters
L= 9
n_s = 10
n_p = 9
t = 1
pi = 3.141592654

#construct basis
basis = boson_basis_1d(L=n_s,Nb=n_p)
dim = len([n for n in basis])
print(f"Dimension = {dim}")
print(basis)

t_ss = [[i,j,k,l] for i in range(n_s) for j in range(n_s) for k in range(n_s) for l in range(n_s)]

for n in t_ss:
    if n[0]+n[1]==n[2]+n[3]:
        n.insert(0,t)
        n[0] = n[0]*(np.exp((-(2*pi**2)/(L**2))*(((n[1]-n[3])**2)+(n[2]-n[3])**2)))
    else:
        n.insert(0,0)

static = [["++--", t_ss]]

H = hamiltonian(static,[],basis = basis, dtype=np.float64, check_symm=False)

matrix = H.toarray()

#Calculate eigenvectors and eigenvalues
eigens = np.linalg.eigh(matrix)

eval_noround= eigens[0]

eval = [float('%.7f' % n) for n in eval_noround]
evect = np.transpose(eigens[1])

print(f"eigenvalues = {eval}")

def groundstate(eval,evect):
    gstate = []
    n = 0
    while eval[n] == eval[0]:
        gstate.append(evect[n])
        n = n+1
    print(f"Degeneracy = {n}")
    return gstate,n

ground,degeneracy = groundstate(eval,evect)


print(f"groundstate = {ground}")

#expectation values

def gaussian(x,k):
    return 1/np.sqrt(pi)*np.exp(-(x-(2*pi*k/L))**2)

gaussian_args = [[i] for i in range(n_s)]

x = np.linspace(-5,15,1000)


den = [[[1,i,i]] for i in range(n_s)]
print(f"order = {den}")

static_den = [["+-",den]]

dynamic = [["+-",den[i],gaussian,gaussian_args[i]] for i in range(n_s)]
print(f"dynamic = {dynamic}")


density = hamiltonian([], dynamic,basis = basis, dtype=np.float64, check_symm=False)

def den_expt(groundstate):
    dens = [0 for i in range(len(x))]
    print(dens)
    for i in range(degeneracy):
        for j in range(len(x)):
            dens[j] = dens[j]+density.expt_value(groundstate[i],time=x[j])
    return dens

dens = den_expt(ground)

plt.plot(x,dens)
print("--- %s seconds ---" % (time.time() - start_time))

plt.show()

