import numpy as np

    
def gap_1D(U):
    
    '''
        The gap between 2 lowest levels in the lowest Hubbard sector for the 1D model
        
        Input: 
        U - Coulomb interaction constant in units of hopping constant
        
        Output:
        scalar, value of the gap 

    '''
    return 2/U

def stagger(x):
    
    '''
    Function that generates a 1D staggered array +1-1+1-1....
    
    Input:
    x - number of lattice sites (size of the array)
    
    Output:
    array itself
    '''
    return np.array([((-1) ** i) for i in range(x)])

def half(x):
    
    '''
    Function that generates a 1D staggered array +1-1+1-1....
    
    Input:
    x - number of lattice sites (size of the array)
    
    Output:
    array itself
    '''
    return np.array([(-1) ** (np.heaviside(i-x//2,1)) for i in range(x)])

def stagger_2D(Lx, Ly):
    
    '''
    Function that generates a 2D staggered array +1-1+1-1....
                                                 -1+1-1+1....
    
    Input:
    Lx, Ly - number of lattice sites in x and y directions (size of the array)
    
    Output:
    array itself
    '''
    
    array = []
    array1 = [((-1) ** i) for i in range(Lx)]
    array2 = [((-1) ** (i+1)) for i in range(Lx)]
    for i in range(Ly):
        if i % 2 == 0:
            array += array1
        elif i % 2 !=0:
            array += array2
    return np.array(array)

def half_2D(Lx, Ly):
    
    '''
    Function that generates a 1D staggered array +1+1+1-1-1-1
                                                 +1+1+1-1-1-1
    
    Input:
    x - number of lattice sites (size of the array)
    
    Output:
    array itself
    '''
    array = np.array([])
    for i in range(Ly):
        array = np.hstack((array, half(Lx)))
    return array

def is_even(p):
    
    '''
    Calculates the sign of the permutation p
    
    Input:
    p - array
    
    Output:
    +1 or -1
    '''
    if len(p) == 1:
        return 1

    trans = 0

    for i in range(0,len(p)):
        j = i + 1

        for j in range(j, len(p)):
            if p[i] > p[j]: 
                trans = trans + 1

    if ((trans % 2) == 0):
        return 1
    elif ((trans % 2)!= 0):
        return -1