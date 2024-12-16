import numpy as np # generic math functions
import math
import matplotlib.pyplot as plt
import os
from time import time

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d, tensor_basis # Hilbert space fermion and tensor bases

def solve_SYK_Hamiltonian(L, Nf, J, mu):
    '''
    Function that solves cSYK_4 Hamiltonian
    '''

    basis = spinless_fermion_basis_1d(L=L,Nf=Nf)

    rnd_couplings = []
    for i in range(L):
        for j in range(i+1, L):
            for k in range(i+1,L):
                for l in range(k+1, L):
                    Jrnd = (np.random.normal(0, J/np.sqrt(2), 1)[0] + 1j*np.random.normal(0, J/np.sqrt(2), 1)[0])/np.power(2*L, 3/2)
                    rnd_couplings.append([Jrnd,i,j,k,l])
                    rnd_couplings.append([-1*Jrnd,j,i,k,l])
                    rnd_couplings.append([-1*Jrnd,i,j,l,k])
                    rnd_couplings.append([Jrnd,j,i,l,k])
                    rnd_couplings.append([np.conj(Jrnd),k,l,i,j])
                    rnd_couplings.append([-1*np.conj(Jrnd),k,l,j,i])
                    rnd_couplings.append([-1*np.conj(Jrnd),l,k,i,j])
                    rnd_couplings.append([np.conj(Jrnd),l,k,j,i])

    for i in range(L):
        for j in range(i+1, L):
            for l in range(j+1, L):
                Jrnd = (np.random.normal(0, J/np.sqrt(2), 1)[0] + 1j*np.random.normal(0, J/np.sqrt(2), 1)[0])/np.power(2*L, 3/2)
                rnd_couplings.append([Jrnd,i,j,i,l])
                rnd_couplings.append([-1*Jrnd,j,i,i,l])
                rnd_couplings.append([-1*Jrnd,i,j,l,i])
                rnd_couplings.append([Jrnd,j,i,l,i])
                rnd_couplings.append([np.conj(Jrnd),i,l,i,j])
                rnd_couplings.append([-1*np.conj(Jrnd),i,l,j,i])
                rnd_couplings.append([-1*np.conj(Jrnd),l,i,i,j])
                rnd_couplings.append([np.conj(Jrnd),l,i,j,i])

    pot = [[-mu,i] for i in range(L)]

    static=[
            ['++--', rnd_couplings], # random 4-fermionic operators
            ['n', pot], # on-site potential
    ]

    dynamic=[]

    # build Hamiltonian
    no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
    H=hamiltonian(static, dynamic, basis=basis, dtype=np.complex64, **no_checks)
    
    E,V=H.eigh();
    
    return basis, E, V


def DOS(E, J, bins_num):
    count, bins, ignored = plt.hist(E/J, bins_num, density=True)
 
    bins = []
    for i in range(bins_num):
        bins.append([])

    Emax = E.max()
    Emin = E.min()
    deltaE = (Emax - Emin)/bins_num

    for energy in E:
        bins[int(energy//deltaE)].append(energy)

    densityE = []
    averagesE = []

    for b in bins:
        densityE.append(len(b)/len(E))
        averagesE.append(sum(b)/len(b)/J)
        
    return densityE, averagesE


