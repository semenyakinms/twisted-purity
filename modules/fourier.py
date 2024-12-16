import numpy as np
import itertools
from modules.functions import *
from time import time

def FT_spinless(basis, vector, L, number_of_electrons):
    
    vector_in_momentum_space = (1+1j) * np.zeros(basis.Ns)
    position_in_vector = 0
    for el in basis:
        index = 0
        state = basis.int_to_state(el)
        state_in_positions = []
        for entry in state[1:-1:2]:
            index += 1
            if int(entry) == 1:
                state_in_positions.append(index)
        #Constructing all the permutations
        for el in list(itertools.permutations(state_in_positions)):
            sign = is_even(el)
            for n in range(len(vector)):
                fourier_matrix = fourier_many_particle_state_spinless(basis.int_to_state(basis[n]), L, number_of_electrons)
                vector_in_momentum_space[position_in_vector] += sign * vector[n] * np.prod([fourier_matrix[i][el[i]-1] for i in range(number_of_electrons)]) 
        position_in_vector += 1 
    return vector_in_momentum_space
 
def FT_spinfull(basis, vector, L, filling=0.5, symmetry=0.5, show_progress=True):
    ti = time()
    number_of_electrons = int(2 * filling * L)
    N_up = int(number_of_electrons * symmetry) # number of fermions with spin up
    N_down = number_of_electrons - N_up # number of fermions with spin down  
    vector_in_momentum_space = (1+1j) * np.zeros(basis.Ns)
    
    position_in_vector = 0
    indic = 0
    for el in basis:
        if show_progress:
            if indic < 2:
                if indic == 1:
                    print("One computation is expected to take {} minutes".format(np.around((time()-ti) * basis.Ns / 60, 1)))
        indic += 1
        if show_progress:
            print('Progress {}'.format(np.around(100 * indic/basis.Ns, 1)), end="\r")
        state = basis.int_to_state(el)
        state_in_positions_up = [pos for pos, entry in enumerate([int(j) for j in state[1:2*(L+1)-1:2]]) if entry == 1]
        state_in_positions_down  = [pos for pos, entry in enumerate([int(j) for j in state[2*(L+1):-1:2]]) if entry == 1]

            #Constructing all the permutations
        for el_up in list(itertools.permutations(state_in_positions_up)):
            for el_down in list(itertools.permutations(state_in_positions_down)):
                sign = is_even(el_up) * is_even(el_down)
                for n in range(len(vector)):
                    fourier_matrix_up, fourier_matrix_down = fourier_many_particle_state_spinfull(basis.int_to_state(basis[n]), L,  filling, symmetry)
                    vector_in_momentum_space[position_in_vector] += sign * vector[n] * np.prod([fourier_matrix_up[i][el_up[i]] for i in range(N_up)]) * np.prod([fourier_matrix_down[i][el_down[i]] for i in range(N_down)]) 
        position_in_vector += 1 
    return vector_in_momentum_space


def fourier_many_particle_state_spinfull(state, L, filling=0.5, symmetry=0.5):
    '''
    Makes a Fourier decomposition of the many particle state |i1 i2....iN>
    '''
    N = int(2 * filling * L)
    N_up = int(N * symmetry) # number of fermions with spin up
    N_down = N - N_up # number of fermions with spin down  

        
    fourier_matrix_up = (1+1j)*np.zeros((N_up, L))
    fourier_matrix_down = (1+1j)*np.zeros((N_down, L))
    index = 0
    electron = 0
    # We count such that the first electon is the furtherst from the vacuum
    for el in state[1:2*L+1-1:2]:
        index += 1
        if int(el) == 1:
            fourier_matrix_up[electron] = fourier_basis(index, L)  
            electron += 1
    index = 0
    electron = 0            
    for el in state[2*(L+1):-1:2]:
        index += 1
        if int(el) == 1:
            fourier_matrix_down[electron] = fourier_basis(index, L)  
            electron += 1
    return fourier_matrix_up, fourier_matrix_down
 
def fourier_many_particle_state_spinless(state, L, number_of_electrons):
    '''
    Makes a Fourier decomposition of the many particle state |i1 i2....iN>
    '''
    
    index = 0
    electron = 0
    fourier_matrix = (1+1j)*np.zeros((number_of_electrons, L))
    # We count such that the first electon is the furtherst from the vacuum
    for el in state[1:-1:2]:
        index += 1
        if int(el) == 1:
            fourier_matrix[electron] = fourier_basis(index, L)  
            electron += 1
    return fourier_matrix

def fourier_many_particle_state_spindown(state, L, number_of_electrons):
    '''
    Makes a Fourier decomposition of the many particle state |i1 i2....iN>
    '''
    
    index = 0
    electron = 0
    fourier_matrix = (1+1j)*np.zeros((number_of_electrons, L))
    # We count such that the first electon is the furtherst from the vacuum
    for el in state[2*(L+1):-1:2]:
        index += 1
        if int(el) == 1:
            fourier_matrix[electron] = fourier_basis(index, L)  
            electron += 1
    return fourier_matrix

def fourier_many_particle_state_spinup(state, L, number_of_electrons):
    '''
    Makes a Fourier decomposition of the many particle state |i1 i2....iN>
    '''
    
    index = 0
    electron = 0
    fourier_matrix = (1+1j)*np.zeros((number_of_electrons, L))
    # We count such that the first electon is the furtherst from the vacuum
    for el in state[1:2*(L+1)-1:2]:
        index += 1
        if int(el) == 1:
            fourier_matrix[electron] = fourier_basis(index, L)  
            electron += 1
    return fourier_matrix

def fourier_basis(n, L):
    '''
    Makes a Fourier decomposition of the nth site
    '''
    vec_k = []
    for i in range(L):
        vec_k.append(np.sqrt(1/L) * np.exp(2 * np.pi* 1j* i * n / L))
    return vec_k

def fourier(vec, L):
    '''
    Makes a FT for 1-particle state
    Input:
        
    '''
    vec_k = (1+1j) * np.zeros(L)
    for i in range(L):
        vec_k[i] = np.sqrt(1 / L) * np.sum([vec[j] * np.exp(2 * np.pi* 1j* i * j / L) for j in range(L)])
    return vec_k
