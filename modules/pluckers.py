import numpy as np # generic math functions
import itertools
import bisect

from time import time
from sys import getsizeof

from numba import jit

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d, tensor_basis # Hilbert space fermion and tensor bases
from quspin.basis import spinful_fermion_basis_1d

no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)

def HigherCDegenSpinless(psi, basis, k):
    
    norm = 0;
    for I in itertools.combinations(range(basis.L), k):
        for J in itertools.combinations(range(basis.L), k):

            operIJ = [[
                '+'*k + '-'*k,
                [[1] + list(I + J)]
            ]];
            operJI = [[
                '-'*k + '+'*k,
                [[1] + list(I + J)]
            ]];
            #print(operIJ, operJI)
            dynamic=[];
            hIJ = hamiltonian(operIJ,dynamic,basis=basis,dtype=np.float64,**no_checks);
            hJI = hamiltonian(operJI,dynamic,basis=basis,dtype=np.float64,**no_checks);

            for a in psi:
                for b in psi:
                    norm += np.dot(np.conjugate(a), hIJ.dot(b))*np.dot(np.conjugate(b), hJI.dot(a));
                    
    return norm/len(psi)/len(psi);

def HigherCDegen(psi, basis, k):

    norm = 0;
    for Nup in range(k+1):
        Ndown = k - Nup
        #print(Nup, Ndown)
        for Iup in itertools.combinations(range(basis.L), Nup):
            for Idown in itertools.combinations(range(basis.L), Ndown):
                for Jup in itertools.combinations(range(basis.L), Nup):
                    for Jdown in itertools.combinations(range(basis.L), Ndown):

                        operIJ = [[
                            '+'*Nup + '-'*Nup + '|' + '+'*Ndown + '-'*Ndown ,
                            [[1] + list(Iup + Jup + Idown + Jdown)]
                        ]];
                        operJI = [[
                            '-'*Nup + '+'*Nup + '|' + '-'*Ndown + '+'*Ndown ,
                            [[1] + list(Iup + Jup + Idown + Jdown)]
                        ]];
                        #print(operIJ, operJI)
                        dynamic=[];
                        hIJ = hamiltonian(operIJ,dynamic,basis=basis,dtype=np.float64,**no_checks);
                        hJI = hamiltonian(operJI,dynamic,basis=basis,dtype=np.float64,**no_checks);

                        for a in psi:
                            for b in psi:
                                norm += np.dot(np.conjugate(a), hIJ.dot(b))*np.dot(np.conjugate(b), hJI.dot(a));
                    
    return norm/len(psi)/len(psi);

def HigherCSpinless(psi, basis, k):
    
    return HigherCDegenSpinless([psi], basis, k)

def HigherC(psi, basis, k):
    
    return HigherCDegen([psi], basis, k);

def insertion_sgn(list1, list2):
    perms = 0
    j=0
    for i in range(len(list1)):
        while j<len(list2) and list1[i]>list2[j]:
            j = j+1
        perms = perms + j
    return perms

def inds_to_mask(L, inds_list):
    # the inds_list should not be empty - otherwise exception
    i = 0
    mask = ''
    for j in range(len(inds_list)):
        while i<inds_list[j]:
            mask += '0'
            i += 1
        mask += '1'
        i += 1
    mask += '0'*(L-inds_list[-1]-1)
    
    return mask

@jit(nopython=False)
def HigherCbyDefinitionSpinless(psi, basis, k):
    L = basis.L
    
    oper = [['+-',[[1,i,i] for i in range(L)]]]
    dynamic=[];
    h = hamiltonian(oper,dynamic,basis=basis,dtype=np.float64,**no_checks)
    Nf = round(np.linalg.norm(h.dot(psi)))
    tot_sum = 0
    a = 0
    for A in itertools.combinations(range(L), Nf+k):
        for B in itertools.combinations(range(L), Nf-k):
            Irange = []
            tmp_sum = 0
            for i in range(L):
                if (i in A) and not(i in B):
                    Irange.append(i)

            for I in itertools.combinations(Irange, k):
                a += 1
                ind1 = []
                for i in A:
                    if i not in I:
                        ind1.append(i)
                ind2 = list(B + I)
                ind2.sort()
                #print(L, Nf, k, A, B, I, ind1, inds_to_mask(L, ind1))
                tmp_sum += (-1)**(insertion_sgn(I,B)+insertion_sgn(I,ind1))*psi[basis.index(inds_to_mask(L, ind1))]*psi[basis.index(inds_to_mask(L, ind2))]
            tot_sum += np.abs(tmp_sum)**2
    print(a)
    return tot_sum

def PluckersSpinless(n_max, basis, V, method = 'HilbertSpaceDoublingMemoryOptimized', silent = True):
    C = []
    if method == 'MatrixElementsSummation':
        t0 = time()
        t1 = time()
    elif method == 'HilbertSpaceDoublingMemoryOptimized':
        t0 = time()
        
        states = []
        for st in basis.states:
            states.append(basis.int_to_state(st)[1:-1].replace(' ',''))
        Nf = sum([int(i) for i in basis.int_to_state(basis[0])[1:-1].replace(' ','')])
        L = basis.L
        
        basis_doubled = spinless_fermion_basis_1d(L=2*L,Nf=2*Nf)
        basis_fliped = np.flip(basis_doubled._basis)
        Vdoubled = np.zeros(basis_doubled.Ns, dtype=np.complex64)
        
        for i in range(len(states)):
            for j in range(len(states)):
                ind = basis_doubled.Ns - bisect.bisect_right(basis_fliped, int(states[i]+states[j],2))
                Vdoubled[ind] = V[i]*V[j]
        
        oper = [['+-',[[1,i,L+i] for i in range(L)]]] 
        dynamic = []
        Omega = hamiltonian(oper,dynamic,basis=basis_doubled,dtype=np.float64,**no_checks)
        
        vtmp = Vdoubled
        for k in range(n_max+1):
            C.append(np.linalg.norm(vtmp)**2/(np.math.factorial(k)**2))
            vtmp = Omega.dot(vtmp)
        t1 = time()
        if silent == False:
            print("We calculated Pluckers at L={} and Nf={} for {:.2f} seconds using HilbertSpaceDoublingMemoryOptimized method".format(L, Nf, t1-t0))
            print("Size of doubled vector was {:.2f} mb".format(getsizeof(Vdoubled)/1024/1024))
            print("Size of Omega was {:.2f} mb".format((Omega.tocsc().data.nbytes+Omega.tocsc().indptr.nbytes+Omega.tocsc().indices.nbytes)/1024/1024))
            print("Values of Pluckers are:")
            print(C)

    elif method == 'CorrelatorsSummation':
        for k in range(0, n_max+1):
            t1 = time()
            plucker_value = np.real(HigherCSpinless(V, basis, k))
            t2 = time()
            if silent == False:
                print("We calculated {}-Plucker for {:.2f} seconds. Its value is {}".format(k, t2-t1, plucker_value))
            C.append(plucker_value)
        
    return C

def PluckersSpinfull(n_max,  basis, V):
    C = []
    for i in range(0,n_max+1):
        t1 = time()
        plucker_value = np.real(HigherC(V, basis, i))
        t2 = time()
        print("We calculated {} Plucker of {} to be {} for {:.2f} seconds".format(i, n_max, plucker_value, t2-t1))
        C.append(plucker_value)
    return C