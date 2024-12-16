import pickle
import itertools
import numpy as np
from itertools import combinations

#===========================================================================
##======================== computations of \nu
#===========================================================================

mem='misc_data/nus'
with open(mem, 'wb') as m:
    pickle.dump(dict(),m)

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)

    if n > 1:
        factors.append(n)
    return factors

def sub_tau(tau, aa):
    new_tau=[]
    for a in aa:
        new_tau+=[tau[a]]
    return new_tau

def nu(tau):
    tau=tuple(tau)
    with open(mem,'rb') as m:
        nu_mem=pickle.load(m)
    if len(tau)==1:
        return 1
    elif tau in nu_mem:
        return nu_mem[tau]
    else:
#        new_nu=Fraction(0)
        new_nu=(0)
        aa_s=set()
        l=len(tau)
        for size in range(1,l):
            aa_s.update(set(itertools.combinations([i for i in range(0,l)],size)))
        aa_s=list(aa_s)   
        aa_s=[sorted(list(aa)) for aa in aa_s]
        for aa in aa_s:
            bb=list( range(l))
            for a in aa:
                bb.remove(a) 
#            new_nu+=Fraction(((-1)**(sum(tau)+1))*(sum(sub_tau(tau,aa))*((-1)**(sum(sub_tau(tau,aa)))) )*nu(sub_tau(tau,aa))*nu(sub_tau(tau,bb)))/sum(tau)
            new_nu+=(((-1)**(sum(tau)+1))*(sum(sub_tau(tau,aa))*((-1)**(sum(sub_tau(tau,aa)))) )*nu(sub_tau(tau,aa))*nu(sub_tau(tau,bb)))/sum(tau)
        nu_update={tau: new_nu}
        nu_mem.update(nu_update)
        
        with open(mem,'wb') as m:
            pickle.dump(nu_mem,m)
        return new_nu
    
#===========================================================================
#================== Spinless Combinatorics of masks/sets of indices
#===========================================================================

def reorder_opers(list1, list2):
    perms = 0
    j=0
    for i in range(len(list1)):
        while j<len(list2) and list1[i]>list2[j]:
            j = j+1
        perms = perms + j
    return perms

def sort_perms(arr):
    perms = 0
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i]>arr[j]:
                perms += 1
    return perms

def partition(P):
    if len(P) == 1:
        yield [ P ]
        return

    first = P[0]
    for smaller in partition(P[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller

def partition_double(P, Q, k):
    if k==0:
        yield []
        return
    
    # k is the largest size of the subset
    if len(P) == 1:
        yield [ [P, Q] ]
        return

    p = P[0]
    for m, q in enumerate(Q):
        for smaller in partition_double(P[1:], Q[:m] + Q[m+1:], k):
            # insert `first` in each of the subpartition's subsets
            for n, subset in enumerate(smaller):
                if q<min(subset[1]) and len(subset[0])<k:
                    yield smaller[:n] + [[ [p] + subset[0], [q] + subset[1] ]]  + smaller[n+1:]
            # put `first` in its own subset 
            yield [ [ [p], [q] ] ] + smaller
            
def m_double_part(part):
    m = []
    for pair in part:
        m.append(len(pair[0]))
    return m
            
def inds_to_mask(L, inds_list):
    s = np.full(L, b'0')
    s[inds_list] = b'1'    
    return s.tobytes().decode("ascii")

def inds_to_mask_FS(L, Nf, inds_holes, inds_parts):
    s = np.concatenate((np.full(Nf, b'1'), np.full(L-Nf, b'0')))
    if len(inds_holes)>0:
        s[inds_holes] = b'0'
        s[inds_parts] = b'1'
    return s.tobytes().decode("ascii")

#===========================================================================
#==================== Spinless anzats computations
#===========================================================================

def amp_c_by_state(basis, state, P, Q):
    N = np.sum([int(d) for d in basis.int_to_state(basis[0])[1:-1].split(' ')])
    L = basis.L
    
    norm = state[basis.index(inds_to_mask_FS(L, N, [], []))]
    amp = state[basis.index(inds_to_mask_FS(L, N, P, Q))]/norm

    for l in range(1, len(P)):
        for P1 in itertools.combinations(P, l):
            P1b = list(set(P) - set(P1))
            P1b.sort()
            for Q1 in itertools.combinations(Q, l):
                Q1b = list(set(Q) - set(Q1))
                Q1b.sort()
                amp -= np.power(-1, len(P1b)+1+reorder_opers(P1b, P1)+reorder_opers(Q1, Q1b))*(l)/len(P)*state[basis.index(inds_to_mask_FS(L, N, list(P1), list(Q1)))]*state[basis.index(inds_to_mask_FS(L, N, P1b, Q1b))]/norm/norm
    return amp

def anzats_truncation(basis, init_state, k): 
    N = np.sum([int(d) for d in basis.int_to_state(basis[0])[1:-1].split(' ')])
    L = basis.L   
    state = np.zeros(basis.Ns)
    norm = init_state[basis.index(inds_to_mask_FS(L, N, [], []))]
    if k == 0:
        state[basis.index(inds_to_mask_FS(L, N, [], []))] = 1
    else:
        state[basis.index(inds_to_mask_FS(L, N, [], []))] = norm
    
    for n in range(1, min(N, L-N)+1):
        for P in combinations(range(N),n):
            for Q in combinations(range(N, L),n):
                elem = 0
                
                for R in partition_double(list(P), list(Q), k):
                    Ptot = []
                    Qtot = []              
                    for i in range(len(R)):
                        #print(R)
                        Ptot = Ptot + R[len(R) - i-1][0]
                        Qtot = Qtot + R[i][1]
                    
                    numer = 1
                    for pair in R:
                        numer = numer*amp_c_by_state(basis, init_state, pair[0], pair[1])
                    
                    elem += nu(m_double_part(R))*numer*np.power(-1,sort_perms(Ptot)+sort_perms(Qtot))
                    
                state[basis.index(inds_to_mask_FS(L, N, list(P), list(Q)))] = norm*elem
    return state/np.linalg.norm(state)

#===========================================================================
#================== Spinful Combinatorics of masks/sets of indices
#===========================================================================

def inds_to_mask_spin(L, inds_lists):
    s_up = np.full(L, b'0')
    s_down = np.full(L, b'0')
    s_up[inds_lists[0]] = b'1'
    s_down[inds_lists[1]] = b'1'
    return [s_up.tobytes().decode("ascii"),s_down.tobytes().decode("ascii")]

def m_double_part_spin(part):
    m = []
    for four in part:
        m.append(len(four[0])+len(four[2]))
    return m

def inds_to_mask_FS_spin(L, N_up, inds_holes_up, inds_parts_up, N_down, inds_holes_down, inds_parts_down, map_up = None, map_down = None):   
    s_up = np.concatenate((np.full(N_up, b'1'), np.full(L-N_up, b'0')))
    s_down = np.concatenate((np.full(N_down, b'1'), np.full(L-N_down, b'0')))
    if len(inds_holes_up)>0:
        s_up[inds_holes_up] = b'0'
        s_up[inds_parts_up] = b'1'
    if len(inds_holes_down)>0:
        s_down[inds_holes_down] = b'0'
        s_down[inds_parts_down] = b'1'

    if map_up!=None:
        s_up = np.array([s_up[map_up[n]] for n in range(len(s_up))], dtype = '|S1')
    if map_down!=None:
        s_down = np.array([s_down[map_down[n]] for n in range(len(s_down))], dtype = '|S1')
            
    return [s_up.tobytes().decode("ascii"), s_down.tobytes().decode("ascii")]

def partition_double_spin(P_up, Q_up, P_down, Q_down, k):
    if k==0:
        yield []
        return
    
    # k is the largest size of the subset
    if len(P_up)==0 and len(P_down)==1:
        yield [ [[], [], P_down, Q_down] ]
        return

    if len(P_up)==1 and len(P_down)==0:
        yield [ [P_up, Q_up, [], []] ]
        return
    
    if len(P_up)>0:
        p = P_up[0]
        for m, q in enumerate(Q_up):
            for smaller in partition_double_spin(P_up[1:], Q_up[:m] + Q_up[m+1:], P_down, Q_down, k):
                # insert `first` in each of the subpartition's subsets
                for n, subset in enumerate(smaller):
                    if len(subset[0])!=0 and q<min(subset[1]) and len(subset[0]+subset[2])<k:
                        yield smaller[:n] + [[ [p] + subset[0], [q] + subset[1], subset[2], subset[3] ]]  + smaller[n+1:]
                    if len(subset[0])==0 and len(subset[2])<k:
                        yield smaller[:n] + [[ [p] + subset[0], [q] + subset[1], subset[2], subset[3] ]]  + smaller[n+1:]
                # put `first` in its own subset 
                yield [ [ [p], [q], [], [] ] ] + smaller
    elif len(P_down)>0:
        p = P_down[0]
        for m, q in enumerate(Q_down):
            for smaller in partition_double_spin(P_up, Q_up, P_down[1:], Q_down[:m] + Q_down[m+1:], k):
                # insert `first` in each of the subpartition's subsets
                for n, subset in enumerate(smaller):
                    if len(subset[2])!=0 and q<min(subset[3]) and len(subset[0]+subset[2])<k:
                        yield smaller[:n] + [[ subset[0], subset[1], [p] + subset[2], [q] + subset[3] ]]  + smaller[n+1:]
                # put `first` in its own subset 
                yield [ [ [], [], [p], [q] ] ] + smaller

#===========================================================================
#==================== Spinful anzats computations
#===========================================================================

def amp_c_by_state_spin(basis, state, P_up, Q_up, P_down, Q_down, map_up = None, map_down = None):
    N_up = np.sum([int(d) for d in basis.int_to_state(basis[0])[1:-1].split('>|')[0].split(' ')])
    N_down = np.sum([int(d) for d in basis.int_to_state(basis[0])[1:-1].split('>|')[1].split(' ')])
    L = basis.L   

    ind = inds_to_mask_FS_spin(L, N_up, [], [], N_down, [], [], map_up, map_down)
    norm = state[basis.index(ind[0],ind[1])]
    
    ind = inds_to_mask_FS_spin(L, N_up, P_up, Q_up, N_down, P_down, Q_down, map_up, map_down)
    amp = state[basis.index(ind[0],ind[1])]/norm

    for l_up in range(0, len(P_up)+1):
        for l_down in range(0, len(P_down)+1):
            if l_up + l_down != len(P_up) + len(P_down):
                for P1_up in itertools.combinations(P_up, l_up):
                    P1b_up = list(set(P_up) - set(P1_up))
                    P1b_up.sort()
                    for P1_down in itertools.combinations(P_down, l_down):
                        P1b_down = list(set(P_down) - set(P1_down))
                        P1b_down.sort()
                        for Q1_up in itertools.combinations(Q_up, l_up):
                            Q1b_up = list(set(Q_up) - set(Q1_up))
                            Q1b_up.sort()
                            for Q1_down in itertools.combinations(Q_down, l_down):
                                Q1b_down = list(set(Q_down) - set(Q1_down))
                                Q1b_down.sort()
                                sgn = np.power(-1, len(P1b_up)+len(P1b_down)+1+reorder_opers(P1b_up, P1_up)+reorder_opers(Q1_up, Q1b_up)+reorder_opers(P1b_down, P1_down)+reorder_opers(Q1_down, Q1b_down))
                                ind1 = inds_to_mask_FS_spin(L, N_up, list(P1_up), list(Q1_up), N_down, list(P1_down), list(Q1_down), map_up, map_down)
                                ind2 = inds_to_mask_FS_spin(L, N_up, P1b_up, Q1b_up, N_down, P1b_down, Q1b_down, map_up, map_down)
                                amp -= sgn*(l_up+l_down)/(len(P_up)+len(P_down))*state[basis.index(ind1[0], ind1[1])]*state[basis.index(ind2[0], ind2[1])]/norm/norm
    return amp

def anzats_truncation_spin(basis, init_state, k, map_up = None, map_down = None): 
    N_up = np.sum([int(d) for d in basis.int_to_state(basis[0])[1:-1].split('>|')[0].split(' ')])
    N_down = np.sum([int(d) for d in basis.int_to_state(basis[0])[1:-1].split('>|')[1].split(' ')])
    L = basis.L   
    state = np.zeros(basis.Ns, dtype = np.complex128)
    ind0 = inds_to_mask_FS_spin(L, N_up, [], [], N_down, [], [], map_up, map_down)
    norm = init_state[basis.index(ind0[0], ind0[1])]

    for n_up in range(0, min(N_up, L-N_up)+1):
        for n_down in range(0, min(N_down, L-N_down)+1):
            for P_up in combinations(range(N_up),n_up):
                for Q_up in combinations(range(N_up, L),n_up):
                    for P_down in combinations(range(N_down),n_down):
                        for Q_down in combinations(range(N_down, L),n_down):
                            elem = 0
                            for R in partition_double_spin(list(P_up), list(Q_up), list(P_down), list(Q_down), k):
                                Ptot_up = []
                                Qtot_up = []
                                Ptot_down = []
                                Qtot_down = [] 
                                for i in range(len(R)):
                                    Ptot_up = Ptot_up + R[len(R) - i-1][0]
                                    Qtot_up = Qtot_up + R[i][1]                       
                                    Ptot_down = Ptot_down + R[len(R) - i-1][2]
                                    Qtot_down = Qtot_down + R[i][3] 
                            
                                numer = 1
                                for four in R:
                                    numer = numer*amp_c_by_state_spin(basis, init_state, four[0], four[1], four[2], four[3], map_up, map_down)
                                elem += nu(m_double_part_spin(R))*numer*np.power(-1,sort_perms(Ptot_up)+sort_perms(Qtot_up)+sort_perms(Ptot_down)+sort_perms(Qtot_down))
                            ind = inds_to_mask_FS_spin(L, N_up, list(P_up), list(Q_up), N_down, list(P_down), list(Q_down), map_up, map_down)
                            state[basis.index(ind[0], ind[1])] = norm*elem
                          
    #=== it is important that this part stays after the loop, since inside of there 0 value is already wrongly assigned once to this element
    if k == 0:
        state[basis.index(ind0[0], ind0[1])] = 1
    else:
        state[basis.index(ind0[0], ind0[1])] = norm

    return state/np.linalg.norm(state)


