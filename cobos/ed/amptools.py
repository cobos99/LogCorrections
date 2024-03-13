from sunny import analytical_shannon as ashannon
import scipy.sparse as sparse
import numpy as np
import heisembergs
import os

def get_numerical_amps(hamiltonian, filepath=None):
    if not os.path.exists(filepath):
        evals, evecs = sparse.linalg.eigsh(hamiltonian, which="SA", k=1, tol=1e-8)
        amps = np.abs(evecs[:, 0])**2
        Egs = evals[0]
        if filepath is not None:
            filedata = np.zeros(len(amps) + 1)
            filedata[0] = Egs
            filedata[1::] = amps
            np.save(filepath, filedata)
    else:
        filedata = np.load(filepath)
        Egs = filedata[0]
        amps = filedata[1::]
    return Egs, amps

def get_xxz_numerical_max_amps(N_arr, delta, global_neg, return_energy=False, return_max_amp_ind=False, return_cnum=False, comparison_tol=1e-8, spin_penalty_factor=10, print_mode=False, save=True):
    max_prob_num_arr = np.zeros_like(N_arr, dtype=float)
    if return_energy: Egs_num_arr = np.zeros_like(N_arr, dtype=float)
    if return_max_amp_ind: max_amp_inds_arr = np.zeros_like(N_arr)
    if return_cnum: n_max_conf_num_arr = np.zeros_like(N_arr, dtype=float)
    for i, N in enumerate(N_arr):
        if print_mode: print(f"\rNum: N = {N}", end="")
        if save: 
            filename = f"Results/amplitudes_xxz_model_N_{N}_negsign_{global_neg}_delta_{delta:.04f}.npy"
        else:
            filename = ""
        if not os.path.exists(filename):
            H = heisembergs.sparse_xxz_hamiltonian(delta, global_neg, N, spin_penalty_factor)
        else:
            H = None
        Egs, amps = get_numerical_amps(H, filename)
        max_amp_ind = np.argmax(amps)
        max_prob_num_arr[i] = amps[max_amp_ind]
        if return_energy: Egs_num_arr[i] = Egs
        if return_max_amp_ind: max_amp_inds_arr[i] = max_amp_ind
        if return_cnum: n_max_conf_num_arr[i] = np.sum(np.isclose(amps, amps[max_amp_ind], atol=comparison_tol))
    to_return = [np.squeeze(max_prob_num_arr)]
    if return_energy: to_return.append(np.squeeze(Egs_num_arr))
    if return_max_amp_ind: to_return.append(np.squeeze(max_amp_inds_arr))
    if return_cnum: to_return.append(np.squeeze(n_max_conf_num_arr))
    return to_return

def get_xxz_numerical_shannon(N_arr, delta, global_neg, return_energy=False, spin_penalty_factor=10, print_mode=False, save=True):
    shannon_arr = np.zeros_like(N_arr, dtype=float)
    if return_energy: Egs_num_arr = np.zeros_like(N_arr, dtype=float)
    for i, N in enumerate(N_arr):
        if print_mode: print(f"\rNum: N = {N}", end="")
        if save: 
            filename = f"Results/amplitudes_xxz_model_N_{N}_negsign_{global_neg}_delta_{delta:.04f}.npy"
        else:
            filename = None
        if not os.path.exists(filename):
            H = heisembergs.sparse_xxz_hamiltonian(delta, global_neg, N, spin_penalty_factor)
        else:
            H = None
        Egs, amps = get_numerical_amps(H, filename)
        shannon_arr[i] = -np.sum(amps*np.log(amps))
        if return_energy: Egs_num_arr[i] = Egs
    to_return = [np.squeeze(shannon_arr)]
    if return_energy: to_return.append(np.squeeze(Egs_num_arr))
    return to_return

def conf_index_to_str(index, N):
    return f"|{int(index):0{N}b}>"

def get_analytical_xx_max_amps(N_arr, return_energy=False, return_max_amp_ind=False, return_cnum=False, comparison_tol=1e-8, print_mode=False):
    max_prob_num_arr = np.zeros_like(N_arr, dtype=float)
    if return_energy: Egs_num_arr = np.zeros_like(N_arr, dtype=float)
    if return_max_amp_ind: max_amp_ind_arr = np.zeros_like(N_arr)
    if return_cnum: n_max_conf_num_arr = np.zeros_like(N_arr, dtype=float)
    for i, N in enumerate(N_arr):
        if print_mode: print(f"\rAn: N = {N}", end="")
        filename = f"Results/an_amplitudes_xx_model_N_{N}.npy"
        if not os.path.exists(filename):
            confarr = ashannon.probabilities(N)
            amps = [conf.prob for conf in confarr]
            Egs = ashannon.ground_state_energy(N)
            filedata = np.zeros(len(amps) + 1)
            filedata[0] = Egs
            filedata[1::] = amps
            np.save(filename, filedata)
        else:
            filedata = np.load(filename)
            Egs = filedata[0]
            amps = filedata[1::]
        max_amp_ind = np.argmax(amps)
        max_prob_num_arr[i] = amps[max_amp_ind]
        if return_energy: Egs_num_arr[i] = Egs
        if return_max_amp_ind: max_amp_ind_arr[i] = max_amp_ind
        if return_cnum: n_max_conf_num_arr[i] = np.sum(np.isclose(amps, amps[max_amp_ind], atol=comparison_tol))
    to_return = [max_prob_num_arr]
    if return_energy: to_return.append(Egs_num_arr)
    if return_max_amp_ind: to_return.append(max_amp_ind_arr)
    if return_cnum: to_return.append(n_max_conf_num_arr)
    return np.squeeze(to_return)