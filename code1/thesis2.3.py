#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 18:34:42 2023

@author: zakariam
"""

import numpy as np

c = 1.6e-19
thlist = [3.1, 7.1, 2.1, 14.2]
x = 0.303
Ecn = [x, 0, x, 0]
me = 9.1093837e-31
mcn = [0.0919, 0.067, 0.0919, 0.067] 
p = 22.67

d = sum(thlist)*1e-9
q = np.pi/d
hbar = 1.05e-34

C = np.sqrt(me*c)/ hbar


#%%




def Valence_energy(Ec, mc):#function to find the valence energy for each layer
    return Ec - (p*mc)


def calc(E):
    matrices = np.zeros((4, 2, 2), dtype=np.complex128) 
    for i in range(len(Ecn)):
        Ev1 = Valence_energy(Ecn[i], mcn[i])
        Ev2 = Valence_energy(Ecn[i-1], mcn[i-1])
        m1 = ((E - Ev1))/(p)
        m2 = ((E - Ev2))/(p)
        Ec1 = Ecn[i]
        Ec2 = Ecn[i-1]
        l1 = np.sqrt(2*m1*(Ec1 - E) + 0j)*C
        l2 = np.sqrt(2*m2*(Ec2 - E) + 0j)*C
        #if l2 == 0 or m1 == 0 or l1 == 0:
           # print(f"m1={m1}, l1={l1} , l2={l2}, m2={m2}")
            #return None
        #else:
        a = (m2*l1)/(m1*l2) 
        b = thlist[i] * 1e-9
        M_i =  0.5 * np.array([[np.exp(l1*b)*(1 + a), np.exp(-l1*b)*(1 - a) ],
                                    [np.exp(l1*b)*(1 - a), np.exp(-l1*b)*(1 + a)]])
        M_i = np.squeeze(M_i)
        matrices[i] = M_i
    return np.prod(matrices, axis=0)
            

def eigenvalue(q, d):#bloch eigenvalue
    return np.exp(1j*q*d)



f = eigenvalue(q, d)

# Define the function to be zeroed 
def det_func(E):
    return np.linalg.det(calc(E) - (np.eye(2)*f))

#%%

from scipy.optimize import root_scalar

# Define the bounds of the energy search interval
E_min = 0.001
E_max = 0.011

# Find the first 5 roots using root_scalar
Roots = []
while len(Roots) < 5:
    if np.sign(det_func(E_min)) != np.sign(det_func(E_max)):    
        sol = root_scalar(det_func, bracket=[E_min, E_max], method='brentq')
        Roots.append(sol.root)
        E_min = sol.root + 0.01
        E_max = E_min + 0.01
    else:
        E_min = E_max
        E_max += 0.01

# Print the roots
print(Roots)

#%%



# Define the bounds of the energy search interval
E_min = 0.001
E_max = 0.011

# Find the first 5 roots using root_scalar
Roots = []
while len(Roots) < 5:
    if np.sign(det_func(E_min)) != np.sign(det_func(E_max)):    
        sol = root_scalar(det_func, bracket=[E_min, E_max], method='brentq')
        if sol.converged:
            Roots.append(sol.root)
        E_min = sol.root + 0.01
        E_max = E_min + 0.01
    else:
        E_min = E_max
        E_max += 0.01

# Print the roots
print(Roots)







#%%
import matplotlib.pyplot as plt


x = np.linspace(0.001, 0.29, 1000)

y = []

for i in range(len(x)):
    y.append(np.real(det_func(x[i])))
    
plt.plot(x, y)
    

#%%

v = np.zeros(1000)

for i in Roots:
    plt.scatter(i, np.real(det_func(i)))

plt.plot(x, y)    
plt.plot(x, v)












