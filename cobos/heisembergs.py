import numpy as np
import scipy.sparse as sparse

# ---------------------------------------------------------------------
# AUXILIARY TOOLS TO SIMULATE THE 1D HEISEMBERG MODEL
# ---------------------------------------------------------------------

def sparse_non_diag_paulis_indices(n, N):
    """Returns a tuple (row_indices, col_indices) containing the row and col indices of the non_zero elements
       of the tensor product of a non diagonal pauli matrix (x, y) acting over a single qubit in a Hilbert
       space of N qubits"""
    if 0 <= n < N:
        block_length = 2**(N - n - 1)
        nblocks = 2**n
        ndiag_elements = block_length*nblocks
        k = np.arange(ndiag_elements, dtype=int)
        red_row_col_ind = (k % block_length) + 2*(k // block_length)*block_length
        upper_diag_row_indices = red_row_col_ind
        upper_diag_col_indices = block_length + red_row_col_ind
        row_indices = np.concatenate((upper_diag_row_indices, upper_diag_col_indices))
        col_indices = np.concatenate((upper_diag_col_indices, upper_diag_row_indices))
        return row_indices, col_indices
    else:
        raise ValueError("Index n must fulfill 0 <= n < N")

def sparse_pauli_x(n, N, row_indices_cache=None, col_indices_cache=None):
    """Returns a CSC sparse matrix representation of the pauli_x matrix acting over qubit n in a Hilbert space of N qubits
       0 <= n < N"""
    if 0 <= n < N:
        if (row_indices_cache is None) or (col_indices_cache is None):
            row_indices_cache, col_indices_cache = sparse_non_diag_paulis_indices(n, N)
        data = np.ones_like(row_indices_cache)
        result = sparse.csc_array((data, (row_indices_cache, col_indices_cache)), shape=(2**N, 2**N), dtype=complex)
        return result
    else:
        raise ValueError("Index n must fulfill 0 <= n < N")

def sparse_pauli_y(n, N, row_indices_cache=None, col_indices_cache=None):
    """Returns a CSC sparse matrix representation of the pauli_y matrix acting over qubit n in a Hilbert space of N qubits
       0 <= n < N"""
    if 0 <= n < N :
        if (row_indices_cache is None) or (col_indices_cache is None):
            row_indices_cache, col_indices_cache = sparse_non_diag_paulis_indices(n, N)
        data = -1j*np.ones_like(row_indices_cache)
        data[len(data)//2::] = 1j
        result = sparse.csc_array((data, (row_indices_cache, col_indices_cache)), shape=(2**N, 2**N), dtype=complex)
        return result
    else:
        raise ValueError("Index n must fulfill 0 <= n < N")

def sparse_pauli_z(n, N):
    """Returns a CSC sparse matrix representation of the pauli_z matrix acting over qubit n in a Hilbert space of N qubits
       0 <= n < N"""
    if 0 <= n < N:
        block_length = 2**(N - n)
        nblocks = 2**n
        block = np.ones(block_length, dtype=int)
        block[block_length//2::] = -1
        diag = np.tile(block, nblocks)
        row_col_indices = np.arange(2**N, dtype=int)
        result = sparse.csc_array((diag, (row_col_indices, row_col_indices)), shape=(2**N, 2**N), dtype=complex)
        return result
    else:
        raise ValueError("Index n must fulfill 0 <= n < N")

def sparse_heisemberg_hamiltonian(J, N):
    """Returns a sparse representation of the Hamiltonian of the 1D Heisemberg model in a chain of length N
       with periodic boundary conditions (hbar = 1)
       J < 0: Antiferromagnetic case (Unique ground state of total angular momentum S=0)
       J > 0: Ferromagnetic case (N+1-fold degeneracy of the ground state of angular momentum N/2) -> Dicke states for even N"""
    hamiltonian = sparse.csc_array((2**N, 2**N), dtype=complex)
   
    # First sum over the terms containing sigma_x, sigma_y because the non-zero element indices are the same
    # so that this improves performance
    n_row_indices, n_col_indices = sparse_non_diag_paulis_indices(0, N)
    for n in range(N):
        ntildep1 = (n+1) % N
        np1_row_indices, np1_col_indices = sparse_non_diag_paulis_indices(ntildep1, N)
        n_pauli_x = sparse_pauli_x(n, N, n_row_indices, n_col_indices)
        np1_pauli_x = sparse_pauli_x(ntildep1, N, np1_row_indices, np1_col_indices)
        n_pauli_y = sparse_pauli_y(n, N, n_row_indices, n_col_indices)
        np1_pauli_y = sparse_pauli_y(ntildep1, N, np1_row_indices, np1_col_indices)
        hamiltonian += (n_pauli_x @ np1_pauli_x) + (n_pauli_y @ np1_pauli_y)
        n_row_indices = np1_row_indices
        n_col_indices = np1_col_indices
   
    # Sum over sigma_z terms
    n_pauli_z = sparse_pauli_z(0, N)
    for n in range(N):
        ntildep1 = (n+1) % N
        np1_pauli_z = sparse_pauli_z(ntildep1, N)
        hamiltonian += n_pauli_z @ np1_pauli_z
        n_pauli_z = np1_pauli_z

    return -J*hamiltonian/4

def sparse_xxz_hamiltonian(delta, antiferro, N, spin_sector_penalty_factor=10):
    """Returns a sparse representation of the Hamiltonian of the 1D XXZ model in a chain of length N with periodic boundary conditions
       
       Arguments
       ---------
       delta: Constant in front of the ZZ term
       antiferro: Boolean setting whether to return the antiferromagnetic Hamiltonian. Sets the sign in front of the (XX + YY) term
       N: Chain length
    """
    hamiltonian = sparse.csc_array((2**N, 2**N), dtype=complex)
    # Ladder operator terms
    n_row_indices, n_col_indices = sparse_non_diag_paulis_indices(0, N)
    sign = (-1)**(antiferro <= 0)
    for n in range(N):
        ntildep1 = (n + 1) % N
        np1_row_indices, np1_col_indices = sparse_non_diag_paulis_indices(ntildep1, N)
        n_pauli_x = sparse_pauli_x(n, N, n_row_indices, n_col_indices)
        np1_pauli_x = sparse_pauli_x(ntildep1, N, np1_row_indices, np1_col_indices)
        n_pauli_y = sparse_pauli_y(n, N, n_row_indices, n_col_indices)
        np1_pauli_y = sparse_pauli_y(ntildep1, N, np1_row_indices, np1_col_indices)
        hamiltonian += sign*((n_pauli_x @ np1_pauli_x) + (n_pauli_y @ np1_pauli_y))
        n_row_indices = np1_row_indices
        n_col_indices = np1_col_indices
    
    # ZZ and penalty terms
    n_pauli_z = sparse_pauli_z(0, N)
    SZ_sum = sparse.csc_array((2**N, 2**N), dtype=complex)
    for n in range(N):
        ntildep1 = (n+1) % N
        np1_pauli_z = sparse_pauli_z(ntildep1, N)
        hamiltonian += delta*(n_pauli_z @ np1_pauli_z)
        SZ_sum += n_pauli_z
        n_pauli_z = np1_pauli_z
    
    penalty_term = (SZ_sum/2 + sparse.identity(hamiltonian.shape[0])/2)
    penalty_term = penalty_term @ penalty_term
    hamiltonian += spin_sector_penalty_factor*penalty_term

    return hamiltonian

def singlet_chain_state(N):
    """Returns the state corresponding to pairs of spin singlet states in a spin chain of N (even) qubits"""
    if (N % 2) == 0:
        power_sum_table = np.array([list(f"{i:0{N//2}b}") for i in range(2**(N//2))], dtype=int)
        even_powers_table = np.ones((2**(N//2), N//2), dtype=int)*np.arange(0, N,2)[::-1]
        to_pow_table = 2*np.ones(shape=(2**(N//2), N//2), dtype=int)
        signs = (-1)**(power_sum_table.sum(axis=1) % 2)
        non_zero_indices = np.sum(to_pow_table**(even_powers_table + power_sum_table), axis=1)
        elements = np.ones(2**(N//2))/2**(N/4)
        elements = signs*elements
        result = np.zeros(2**N, dtype=complex)
        result[non_zero_indices] = elements
        return result
    else:
        raise ValueError("The singlet chain state can only be defined in even-length lattices")
    
def uniform_state(N):
    unnorm_vec = np.ones(2**N)
    return unnorm_vec / np.linalg.norm(unnorm_vec)
