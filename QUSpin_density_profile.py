from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
import numpy as np
import matplotlib.pyplot as plt

import time
start_time = time.time()

#Parameters
L= 3
n_s = 5
n_p = 27
t = 1
pi = 3.141592654

#construct basis
basis = boson_basis_1d(L=n_s,Nb=n_p)
dim = len([n for n in basis])
print(f"Dimension = {dim}")

#Construct hamiltonian as in Kovrizhin 2010

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
evect = eigens[1]


#Convert basis from QUSpin format to a list

def basistolist(basis):
  string_basis = [basis.int_to_state(n) for n in basis]
  string_list = [list(n) for n in string_basis]

  basis_list = []
  for n in string_list:
    n.remove('|')
    n.remove('>')
    while n.count(' ') != 0:
      n.remove(' ')

    basis_element = [int(i) for i in n]
    basis_list.append(basis_element)
  return basis_list

basis_list = basistolist(basis)


#Convert from occupation number representation to Landau wave-function

def occupation(eval, evect):
  occupation = [0 for n in range(n_s)]

  state = 0

  while eval[0]==eval[state]:
    groundstate_degen = np.matrix(evect.transpose()[state])
    groundstate_degen = groundstate_degen.tolist()[0]
    occupation_degen = []

    for n in range(n_s):
      temp = 0
      for m in range(len(basis_list)):
        temp = temp + ((groundstate_degen[m]**2)*basis_list[m][n])
      occupation_degen.append(temp)

    occupation = [occupation[n]+occupation_degen[n] for n in range(n_s)]

    state = state + 1
  occupation = [occupation[n]/state for n in range(n_s)]
  print(state,"-fold Degenerate")
  return occupation

#Plot the density profile

y = np.arange(-5,25.1,0.1)

particles = occupation(eval,evect)
print(particles)
density = []

for m in y:
  temp = 0
  for n in range(n_s):
    temp = temp+ (1/np.sqrt(pi))*particles[n]*np.exp(-(m-(2*pi*n/L))**2)
  density.append(temp)

plt.plot(y,density)
plt.xlabel("$y/l_b$")
plt.ylabel("Density")

print("--- %s seconds ---" % (time.time() - start_time))

plt.show()





