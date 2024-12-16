from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinful_fermion_basis_1d, spinless_fermion_basis_1d # Hilbert spaces
import numpy as np # generic math functions


no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)


##############################################################
## 1D spinfull Hubbard model ##############
##############################################################

def make_hamiltonian(L, U, B, mu=0, filling=0.5, J=1, symmetry=0.5):
    '''
        Constructs the hamiltonian ща 1D Hubbard model and the basis for the system

        Input:
        L - system size
        U - Coulomb interaction constant in units of hopping constant
        B - array of coordinate-dependent magnetic field
        mu - chemical potential
        filling - specifies the filling. 
        J - hopping constant, by default set to one which means that everything is measures in terms of it
        symmetric - N_up=N_down

        Output:

        Array consisting of quspin objects, first element is basis and second is the Hamiltonian itself 

    '''

    N = int(2 * filling * L)
    N_up = int(N * symmetry) # number of fermions with spin up
    N_down = N - N_up # number of fermions with spin down
    ###### create the basis
    # build spinful fermions basis
    basis = spinful_fermion_basis_1d(L,Nf=(N_up,N_down))

    ##### create model

    # define site-coupling lists
    hop_right=[[-J,i,(i+1)%L] for i in range(L)] #PBC
    hop_left= [[+J,i,(i+1)%L] for i in range(L)] #PBC 
    #hop_right=[[-J,i,(i+1)] for i in range(L-1)] #OBC
    #hop_left= [[+J,i,(i+1)] for i in range(L-1)] #OBC 
    pot_up = [[-mu + B[i], i] for i in range(L)] # -\mu + B \sum_j n_{j \sigma}
    pot_down = [[-mu - B[i], i] for i in range(L)] # -\mu - B \sum_j n_{j \sigma}
    interact = [[U,i,i] for i in range(L)] # U/2 \sum_j n_{j,up} n_{j,down}

    # define static and dynamic lists
    static=[
            ['+-|',hop_left],  # up hops left
            ['-+|',hop_right], # up hops right
            ['|+-',hop_left],  # down hops left
            ['|-+',hop_right], # down hops right
            ['n|',pot_up],        # up on-site potention
            ['|n',pot_down],        # down on-site potention
            ['n|n',interact]   # up-down interaction
                                    ]
    dynamic=[]

    # build Hamiltonian
    H=hamiltonian(static,dynamic,N=L,basis=basis,dtype=np.float64,**no_checks)    
    
    return [basis, H]


def eigensystem(L, U, B, mu=0, filling=0.5, J=1, symmetry=0.5):
    
    '''
        Calculates the eigenvectors and eigenvalues of the 1D Hubbard model

        Input:
        L - system size
        U - Coulomb interaction constant in units of hopping constant
        B - array of coordinate-dependent magnetic field
        mu - chemical potential
        filling - specifies the filling. 
        J - hopping constant, by default set to one which means that everything is measures in terms of it
        symmetric - N_up=N_down

        Output:

        Array, first element is basis, second element is a tuple eigenvalues, eigenvectors 

    '''

    basis, H = make_hamiltonian(L, U, B, mu, filling, J, symmetry)
    
    return [basis, H.eigh()]


def eigensystem_k_lowest(L, U, B,  k, mu=0, filling=0.5, J=1, symmetry=0.5):
    
    '''
        Calculates eigenvectors and eigenvalues of the system, but only the ones with k lowest energy levels

        Input:
        L - system size
        U - Coulomb interaction constant in units of hopping constant
        B - array of coordinate-dependent magnetic field
        mu - chemical potential
        filling - specifies the filling. 
        J - hopping constant, by default set to one which means that everything is measures in terms of it
        symmetric - N_up=N_down

        Output:

        Array, first element is basis, second element is a tuple eigenvalues, eigenvectors 

    '''
    
    basis, H = make_hamiltonian(L, U, B, mu, filling, J, symmetry)
    
    return [basis, H.eigsh(k=k,which="SA",maxiter=1E4,return_eigenvectors=True)]


def ground_state(L, U, B, mu=0, filling=0.5, J=1, symmetry=0.5):
    
    '''
        Calculates ground state and its energy. Does not account for the fact that the GS can be degenearte
        Input:
        L - system size
        U - Coulomb interaction constant in units of hopping constant
        B - array of coordinate-dependent magnetic field
        mu - chemical potential
        filling - specifies the filling. 
        J - hopping constant, by default set to one which means that everything is measures in terms of it
        symmetric - N_up=N_down

        Output:

        Array, first element is basis, second element is ground state energy, third - ground state vector 

    '''

    
    basis, H = make_hamiltonian(L, U, B, mu, filling, J, symmetry)
    E, vectors = H.eigsh(k=1, which="SA", maxiter=1E4, return_eigenvectors=True)
    ground_state_vector = np.zeros(basis.Ns)
    i = 0
    for el in vectors:
        ground_state_vector[i] = el[0]
        i += 1
    
    return [basis, E[0], ground_state_vector]