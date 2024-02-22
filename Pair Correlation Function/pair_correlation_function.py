from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
import numpy as np
import matplotlib.pyplot as plt

import time
start_time = time.time()

#Parameters
L= 4
n_s = 9
n_p = 3
t = 1
pi = 3.141592654
x0 = 0
#construct basis
basis = boson_basis_1d(L=n_s,Nb=n_p)
dim = len([n for n in basis])
print(f"Dimension = {dim}")


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

eval_noround = eigens[0]

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

#expectation values

def gaussian(x,k1,k2,k3,k4):
    return (1/(np.sqrt(pi)**2))*np.exp(-((x0-(2*pi*k1/L))**2)/2)*np.exp(-((x-(2*pi*k2/L))**2)/2)*np.exp(-((x0-(2*pi*k3/L))**2)/2)*np.exp(-((x-(2*pi*k4/L))**2)/2)


temp_gaussian_arg = [[i,j,k,l] for i in range(n_s) for j in range(n_s) for k in range(n_s) for l in range(n_s)]

gaussian_args = []

for i in temp_gaussian_arg:
    if i[0]+i[1] == i[2]+i[3]:
        gaussian_args.append(i)


x = np.linspace(-5,15,1000)

temp_corr = [[[1,i,j,k,l]] for i in range(n_s) for j in range(n_s) for k in range(n_s) for l in range(n_s)]

corr = []

for i in temp_corr:
    if i[0][1]+i[0][2] == i[0][3]+i[0][4]:
        corr.append(i)

dynamic = [["++--",corr[i],gaussian,gaussian_args[i]] for i in range(n_s)]

correlation = hamiltonian([], dynamic,basis = basis, dtype=np.float64, check_symm=False, check_herm=False)

def corr_expt(groundstate):
    corrs = [0 for i in range(len(x))]
    for i in range(degeneracy):
        for j in range(len(x)):
            corrs[j] = corrs[j]+correlation.expt_value(groundstate[i],time=x[j])
    return corrs

dens = corr_expt(ground)

plt.plot(x,dens)
print("--- %s seconds ---" % (time.time() - start_time))


plt.show()