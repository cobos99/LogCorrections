import scipy.sparse as sparse
import numpy as np
import paulis

# ---------------------------------------------------------------------
# AUXILIARY TOOLS TO SIMULATE THE 1D HEISEMBERG MODEL
# ---------------------------------------------------------------------

def sparse_heisemberg_hamiltonian(J, N):
    """Returns a sparse representation of the Hamiltonian of the 1D Heisemberg model in a chain of length N
       with periodic boundary conditions (hbar = 1)
       J < 0: Antiferromagnetic case (Unique ground state of total angular momentum S=0)
       J > 0: Ferromagnetic case (N+1-fold degeneracy of the ground state of angular momentum N/2) -> Dicke states for even N"""
    hamiltonian = sparse.csc_array((2**N, 2**N), dtype=complex)
   
    # First sum over the terms containing sigma_x, sigma_y because the non-zero element indices are the same
    # so that this improves performance
    n_row_indices, n_col_indices = paulis.sparse_non_diag_paulis_indices(0, N)
    for n in range(N):
        ntildep1 = (n+1) % N
        np1_row_indices, np1_col_indices = paulis.sparse_non_diag_paulis_indices(ntildep1, N)
        n_pauli_x = paulis.sparse_pauli_x(n, N, n_row_indices, n_col_indices)
        np1_pauli_x = paulis.sparse_pauli_x(ntildep1, N, np1_row_indices, np1_col_indices)
        n_pauli_y = paulis.sparse_pauli_y(n, N, n_row_indices, n_col_indices)
        np1_pauli_y = paulis.sparse_pauli_y(ntildep1, N, np1_row_indices, np1_col_indices)
        hamiltonian += (n_pauli_x @ np1_pauli_x) + (n_pauli_y @ np1_pauli_y)
        n_row_indices = np1_row_indices
        n_col_indices = np1_col_indices
   
    # Sum over sigma_z terms
    n_pauli_z = paulis.sparse_pauli_z(0, N)
    for n in range(N):
        ntildep1 = (n+1) % N
        np1_pauli_z = paulis.sparse_pauli_z(ntildep1, N)
        hamiltonian += n_pauli_z @ np1_pauli_z
        n_pauli_z = np1_pauli_z

    return -J*hamiltonian/4

def sparse_xxz_hamiltonian(delta, global_neg, N, spin_sector_penalty_factor=10):
    """Returns a sparse representation of the Hamiltonian of the 1D XXZ model in a chain of length N with periodic boundary conditions
       
       Arguments
       ---------
       delta: Constant in front of the ZZ term
       global_neg: Boolean setting whether a minus sign is introduced in fron of the Hamiltonian
       N: Chain length
    """
    hamiltonian = sparse.csc_array((2**N, 2**N), dtype=complex)
    # Ladder operator terms
    n_row_indices, n_col_indices = paulis.sparse_non_diag_paulis_indices(0, N)
    sign = (-1)**(global_neg)
    for n in range(N):
        ntildep1 = (n + 1) % N
        np1_row_indices, np1_col_indices = paulis.sparse_non_diag_paulis_indices(ntildep1, N)
        n_pauli_x = paulis.sparse_pauli_x(n, N, n_row_indices, n_col_indices)
        np1_pauli_x = paulis.sparse_pauli_x(ntildep1, N, np1_row_indices, np1_col_indices)
        n_pauli_y = paulis.sparse_pauli_y(n, N, n_row_indices, n_col_indices)
        np1_pauli_y = paulis.sparse_pauli_y(ntildep1, N, np1_row_indices, np1_col_indices)
        hamiltonian += sign*((n_pauli_x @ np1_pauli_x) + (n_pauli_y @ np1_pauli_y))
        n_row_indices = np1_row_indices
        n_col_indices = np1_col_indices
    
    # ZZ and penalty terms
    n_pauli_z = paulis.sparse_pauli_z(0, N)
    if N % 2 != 0: SZ_sum = sparse.csc_array((2**N, 2**N), dtype=complex)
    for n in range(N):
        ntildep1 = (n+1) % N
        np1_pauli_z = paulis.sparse_pauli_z(ntildep1, N)
        hamiltonian += sign*delta*(n_pauli_z @ np1_pauli_z)
        if N % 2 != 0: SZ_sum += n_pauli_z
        n_pauli_z = np1_pauli_z
    
    if N % 2 != 0:
        penalty_term = (SZ_sum/2 - sparse.identity(hamiltonian.shape[0])/2)
        penalty_term = penalty_term @ penalty_term
        hamiltonian += spin_sector_penalty_factor*penalty_term

    return hamiltonian

def local_thermal_current_operator(delta, n, N):
    """Returns the local thermal current operator in sparse representation"""
    if 0 <= n < N:
        np_tilde = (n+1) % N
        npp_tilde = (n+2) % N
        nm_tilde = (n-1) % N
        first_term = paulis.sparse_pauli_z(n, N) @ (paulis.sparse_ladder_inc(nm_tilde, N) @ paulis.sparse_ladder_dec(np_tilde, N) - paulis.sparse_ladder_inc(np_tilde, N) @ paulis.sparse_ladder_dec(nm_tilde, N))
        second_term = (paulis.sparse_pauli_z(nm_tilde, N) + paulis.sparse_pauli_z(npp_tilde, N))*(paulis.sparse_ladder_inc(n, N) @ paulis.sparse_ladder_dec(np_tilde, N) - paulis.sparse_ladder_inc(np_tilde, N) @ paulis.sparse_ladder_dec(n, N))
        return -1j*(first_term - delta*second_term)
    else:
        raise ValueError("Index n must fulfill 0 <= n < N")
    
def total_thermal_current_operator(delta, N):
    """Returns the total thermal current operator in sparse representation"""
    total_operator = sparse.csc_array((2**N, 2**N), dtype=complex)
    for n in range(N):
        total_operator += local_thermal_current_operator(delta, n, N)
    return total_operator

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
