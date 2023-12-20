#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 18:34:42 2023

@author: zakariam
"""
import numpy as np
from scipy.optimize import root_scalar
import scipy.constants as cd
import matplotlib.pyplot as plt
import pandas as pd


#%%
# constants

c = cd.e # electric charge
me = cd.m_e# mass of electron
hbar = cd.hbar 
C = np.sqrt(me*c)* 1e-9 / hbar 

#%%

#input variables, in final version these will be empty lists, appended with values from an input file

thlist = np.array([3.1, 7.1, 2.1, 14.2]) # thickness of the layers
Ecn = np.array([0.303, 0, 0.303, 0])# conduction band offset
mcn = np.array([0.0919, 0.067, 0.0919, 0.067])# eff mass, in me
p = 22.67# Eg/meff
d = sum(thlist)# 1e-9 # thickness of module
Nq = 900 #Number of q values
Ev = Ecn - (p*mcn)#Valence band offset
Nr = 7#Number of bands
Nz = 300


#%%



def lambdar(E):# function to calculate the effective mass and lambda for each layer
    m = (E- Ev)/p
    l = np.sqrt(2*m*(Ecn - E) + 0j)*C
    return l, m


def layer_matrices(E):#function to calculate hamiltonian the matrices for each layer
    l, m = lambdar(E)
    a = (np.roll(m,1)/m) * (l/np.roll(l,1))#expression for alpha
    b = thlist
    exp1 = np.exp(l*b)
    exp2 = 1/exp1
    M = 0.5 * np.array([[exp1*(1 + a), exp2*(1 - a) ], #matrix of current layer
                          [exp1*(1 - a), exp2*(1 + a)]])

    M = np.rollaxis(M, 2)#shift axis of the matrix into a more understandble form
    
    return M#returns H-matrix for each layer
    

#%%

def moduleprod(E):
    md = layer_matrices(E)
    result = md[0]
    for matrix in md[1:]:# iterate over matrices, except first matrix
        result = np.matmul(matrix, result)# multiply matrices
    return result#return product(matrix of final layer)


def det_func(E, q):#def det_func(E):# the determinant that must be zeroed
    return np.linalg.det(moduleprod(E) - np.eye(2)*np.exp(1j*q*d))


def rootfinder(q):
    # Define the bounds of the energy search interval
    E_min = 0.001
    E_max = 0.011

    # Find all the roots using root_scalar
    roots = []
    while True:
        if np.sign(det_func(E_min, q)) != np.sign(det_func(E_max, q)):
            sol = root_scalar(det_func, bracket=[E_min, E_max], args=(q,), method='brentq')
            roots.append(sol.root)
            E_min = sol.root + 0.01
            E_max = E_min + 0.01
        else:
            E_min = E_max
            E_max += 0.01

        # Stop if the maximum number of roots is found or if the search interval is too large
        if len(roots) >= Nr:
            break
            
    return roots


#q = np.linspace(-np.pi/d, np.pi/d, Nq)# Define the range of q values 
q = np.linspace(-np.pi/d*(1-1/Nq), np.pi/d*(1-1/Nq), Nq)# Define the range of q values as script NEW


def rootsarr(Nq):#function to store the roots for each q value
    roots_arr = np.zeros((Nq, Nr, 3), dtype=np.complex128) # Initialize an array to store the roots for each q value
    
    for i in range(Nq):
        roots = rootfinder(q[i])# Find the roots for the current q value
        for j in range(Nr):

            M = moduleprod(roots[j])# Calculate the module product for the current root
            
            # Obtain co-effiecients A and B from current matrix
            A = -M[0,1]
            B = M[0,0] - np.exp(1j*q[i]*d)
            #print(B)            
            
            # Store the root and its associated values in the roots_arr array
            roots_arr[i,j,:] = np.array([roots[j], A, B])

    return roots_arr

# df is numpy array [iq,inu,sort] sort=0:E sort=1:A,sort=2:B
df = rootsarr(Nq)#make roots_arr global for ease of access to values


#%%

def psi_Cfunc(A, B, lambd, z_n, z_arr):#function to construct conduction band wavefunction
    exper = np.exp(lambd * (z_arr - z_n))
    #here z_n-z_arr gives the distance travelled within a specific layer
    return A*exper + B/exper

def psiV_func(E, Ev, grad):#function to calculate valence band wavefunction
    const = np.sqrt(p*me*c/2)*(hbar/me)*1e9
    return (1/(E - Ev))*(const/c)*grad


z = np.linspace(0, d, 300, endpoint=False)
dz = z[1]-z[0]


def derivative_arr(i, arr):#bespoke derivative calculator with the correct Bloch boundary conditions
    del_arr = (np.roll(arr, -1) - np.roll(arr, 1))/ (2*dz)
    del_arr[0] = (arr[1] - np.exp(-1j*q[i]*d)*arr[-1])/ (2*dz)
    del_arr[-1] = (np.exp(1j*q[i]*d)*arr[0]-arr[-2])/ (2*dz) #NEW
    
    return del_arr
    

def calculate_Bloch(q, v):
    # Define the boundaries of each layer
    bounds = np.zeros(len(thlist)+1)
    bounds[1:] = np.cumsum(thlist)

    # Initialize arrays for the conduction and valence band wavefunctions
    psiC_arr = np.zeros(len(z), dtype=np.complex128)
    psiV_arr = np.zeros(len(z), dtype=np.complex128)

    # Get the coefficients for the current energy eigenvalue and wavevector
    coeff = df[q,v,:]
    E = coeff[0]
    AB = coeff[1:].reshape((2,1))

    # Obtain Lamba and layer matricies for the given energy
    lambd, _ = lambdar(E)
    M = layer_matrices(E)

    ev_arr = np.zeros(len(z))
    for i in range(len(bounds)-1):
        # Mask to select the points in the current layer
        mask = (z >= bounds[i]) & (z < bounds[i+1])
        ev_arr[mask]=Ev[i]   #NEW

        # Calculate the conduction band wavefunction for the current layer
        psiC_arr[mask] = psi_Cfunc(AB[0], AB[1], lambd[i], bounds[i], z[mask])

        # Update the coefficients for the next layer
        AB = M[i] @ AB

    # Calculate the derivative of the conduction band wavefunction
    del_arr = derivative_arr(q, psiC_arr)

    # Calculate the valence band wavefunction for each point in the z-direction
    for x in range(len(z)):
        psiV_arr[x] = psiV_func(E, ev_arr[x], del_arr[x])
        #psiV_arr.append(psiV_func(E, ev_arr[x], del_arr[x]))

    # Convert the list of valence band wavefunction values to a numpy array
    #psiV_arr = np.array(psiV_arr, dtype=np.complex128)

    # Normalize the wavefunctions
    norm_const = sum((abs(psiC_arr)**2 + abs(psiV_arr)**2)*dz)
    psiC_arr, psiV_arr = psiC_arr/np.sqrt(norm_const), psiV_arr/np.sqrt(norm_const)

    # Return the normalised conduction and valence wave-functions array
    return np.array([psiC_arr, psiV_arr])

    
def Total_Bloch(v):
    Bloch_arr = np.zeros((Nq, 2, Nz), dtype = np.complex128) #initialize array to hold the Bloch wavefunctions for all q NEW

    for f in range(Nq):# loop over all q-values and calculate the Bloch wavefunctions for a given band v
        Bloch_arr[f,:] = calculate_Bloch(f,v)
        
    return Bloch_arr



def BlochPhase2(arr):
    Xarr = np.zeros(Nq, dtype=np.complex128)
    Harr= np.zeros((2,Nz),dtype=np.complex128)
    Darr= np.zeros((2,Nz),dtype=np.complex128)
    for i in range(Nq):
        for j in range(2):
            Harr[j,:] = arr[i,j,:]*np.exp(-1j*q[i]*z[:])
            if i==0:
                Darr[j,:] = (arr[i+1,j,:]*np.exp(-1j*q[i+1]*z[:])
                           -arr[Nq-1,j,:]*np.exp(-1j*(q[Nq-1]-2*np.pi/d)*z[:]))*Nq*d/2/np.pi
            elif i==Nq-1:
                Darr[j,:] = (arr[0,j,:]*np.exp(-1j*(q[0]+2*np.pi/d)*z[:])
                           -arr[i-1,j,:]*np.exp(-1j*q[i-1]*z[:]))*Nq*d/2/np.pi
            else:
                Darr[j,:] = (arr[i+1,j,:]*np.exp(-1j*q[i+1]*z[:])
                           -arr[i-1,j,:]*np.exp(-1j*q[i-1]*z[:]))*Nq*d/4/np.pi
        
            Xarr[i]=1j*dz*sum(np.conjugate(Harr[0,:])*Darr[0,:]
                             +np.conjugate(Harr[1,:])*Darr[1,:])
 
    xv = (np.sum(Xarr, axis=0))/Nq#take average Xv(q) over all q
    sX = Xarr - xv
    if Nq%2 ==0:
        i0=int(Nq/2)
        sref=sum(sX[0:i0-1])+sX[i0]/2 #Estimate for 0 which is between those two
    else:
        i0=(Nq-1)/2
        sref=sum(sX[0:i0])
    phase = (np.cumsum(sX)-sref)*2*np.pi/(Nq*d)
    
    return phase



def expectationz(n, v):       
    zmax = np.linspace(-n*d, (n+1)*d, (2*n+1)*Nz, endpoint=False) #NEW
    arr = Total_Bloch(v)
    fullarr = np.zeros((Nq, 2, (2*n+1)*Nz), dtype=np.complex128)  #NEW
    exp = [i for i in range(-n, n+1)]
    for i in range(0,2*n+1):
        for j in range(Nq):
            fullarr[j,:,i*Nz:(i+1)*Nz] = arr[j,:,:]*np.exp(1j*exp[i]*q[j]*d)   #NEW

    wannier_func = (np.sum(fullarr, axis=0))/Nq
    #print('here')
    #plt.plot(zmax, np.real(abs(wannier_func[0])**2))
    #plt.show()
    return (sum((abs(wannier_func[0])**2 + abs(wannier_func[1])**2)*dz))
#wannierSimp(3,0)

zmax = np.linspace(-3*d, (3+1)*d, (2*3+1)*Nz, endpoint=False)

def plotter2(n, v, c, k, ax, fr):
    zmax = np.linspace(-n*d, (n+1)*d, (2*n+1)*Nz, endpoint=False)
    arr = Total_Bloch(v)
    fullarr = np.zeros((Nq, 2, (2*n+1)*Nz), dtype=np.complex128)
    exp = [i for i in range(-n, n+1)]
    for i in range(0, 2*n+1):
        for j in range(Nq):
            fullarr[j, :, i*Nz:(i+1)*Nz] = arr[j, :, :] * np.exp(1j*exp[i]*q[j]*d)
    
    phase = BlochPhase2(arr)
    for i in range(Nq):
        for j in range(2):
            fullarr[i, j, :] = fullarr[i, j, :] * np.exp(1j*phase[i])
     
    eg = sum(np.real(df[:,v,0]))/Nq
    wannier_func = (np.sum(fullarr, axis=0))/Nq
    absWannier =  abs(wannier_func[0])**2
    shiftedwannier = (absWannier*(fr/4)) + eg 
    s = np.roll(np.array(['-','--', '--', '--', '--', '--','--']), k)

    ax.plot(zmax, shiftedwannier, color=c, ls= s[0])
    ax.plot(zmax, np.roll(shiftedwannier, Nz), color=c,ls= s[1])
    ax.plot(zmax, np.roll(shiftedwannier, 2*Nz), color=c, ls= s[2])
    ax.plot(zmax, np.roll(shiftedwannier, 3*Nz), color=c, ls= s[3])
    ax.plot(zmax, np.roll(shiftedwannier, -Nz), color=c,ls= s[4])
    ax.plot(zmax, np.roll(shiftedwannier, -2*Nz), color=c, ls= s[5])
    ax.plot(zmax, np.roll(shiftedwannier, -3*Nz), color=c, ls= s[6])
    
# Create a figure and axes
fig, ax = plt.subplots()

bounds = np.zeros(len(thlist)+1)
bounds[1:] = np.cumsum(thlist)

bh = np.zeros(Nz)

for i in range(len(bounds)-1):
    mask = (z >= bounds[i]) & (z < bounds[i+1])
    bh[mask] = Ecn[i]

qcl = np.tile(bh, 7)


factors = np.array([1, 1, 1, 1, 1, 1])
colors = ['#6A5ACD', '#40E0D0', '#8B4513', '#FA8072', '#DDA0DD', '#FFD700']

#for i in range(Nr-1):
    #plotter2(3, i, colors[i], i, ax, factors[i])
ax.set_xlim(-d, 2*d)
ax.plot(zmax, qcl, 'black', linewidth = '0.5')   
plotter2(3, 0, colors[0], 0, ax, factors[0])
plotter2(3, 1, colors[1], 0, ax, factors[1])
plotter2(3, 2, colors[2], 0, ax, factors[2])
plotter2(3, 3, colors[3], 0, ax, factors[3])
plotter2(3, 4, colors[4], 0, ax, factors[4])
plotter2(3, 5, colors[5], 0, ax, factors[5])
plt.xlabel('Growth Direction (nm)')
plt.ylabel('Energy (eV)')


plt.show()




#%%
















