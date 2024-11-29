import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import os

def min_entropy_delta_L_scaling_plots(max_probabilities_matrix, L_range, delta_range, reg_p0s=[0.1, 0.1, 0.1], filepath=""):
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{physics} \usepackage{mathtools}")
    plt.rc("font", family="serif", size=22, weight="bold")

    # Perform the regressions for each delta
    L_arr = np.arange(L_range[0], L_range[1]+1, max_probabilities_matrix.shape[0])
    delta_arr = np.linspace(*delta_range, max_probabilities_matrix.shape[1])
    min_entropy_func = lambda L, a, b, c: a*L + b*np.log(L) + c
    min_entropy_matrix = -np.log(max_probabilities_matrix)
    as_arr = np.zeros_like(delta_arr)
    a_errors_arr = np.zeros_like(delta_arr)
    bs_arr = np.zeros_like(delta_arr)
    b_errors_arr = np.zeros_like(delta_arr)
    cs_arr = np.zeros_like(delta_arr)
    c_errors_arr = np.zeros_like(delta_arr)
    for i, delta in enumerate(delta_arr):
        regression = sp.optimize.curve_fit(min_entropy_func, L_arr, min_entropy_matrix[:, i], p0=reg_p0s)
        as_arr[i], bs_arr[i], cs_arr[i] = regression[0]
        a_errors_arr[i], b_errors_arr[i], c_errors_arr[0] = np.sqrt(np.diag(regression[1]))

    # Plot each regression variable for different values of delta
    y_plot_dict = {"a": as_arr, "b":bs_arr, "c":cs_arr}
    y_err_dict = {"a":a_errors_arr, "b":b_errors_arr, "c":c_errors_arr}
    color = plt.get_cmap("Set2")((2 % 8)/ 8 + 0.01)
    for varname in ["a", "b", "c"]:
        fig, ax = plt.subplots(figsize=[9, 6])
        plt.errorbar(delta_arr, y_plot_dict[varname], y_err_dict[varname], marker="^", markersize=11, markeredgecolor="black", color=color, zorder=5)
        plt.xlabel(r"$\Delta$", labelpad=10)
        plt.ylabel(rf"${varname}$", labelpad=10)
        plt.title(r"$S_{\infty} = a L + b \log L + c$", fontsize=18)
        xticks = np.linspace(delta_arr[0], delta_arr[-1], 5)
        plt.xticks(xticks, [rf"{tick:.1f}" for tick in xticks])
        plt.ylim([np.min(y_plot_dict[varname]) - 0.05*(np.max(y_plot_dict[varname]) - np.min(y_plot_dict[varname])), np.max(y_plot_dict[varname]) + 0.05*(np.max(y_plot_dict[varname]) - np.min(y_plot_dict[varname]))])
        plt.grid(plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0))
        plt.tight_layout()
        if filepath:
            this_filename = "".join(os.path.basename(filepath).split(".")[:-1])
            extension = os.path.basename(filepath).split(".")[-1]
            this_folder = os.path.dirname(filepath)
            this_filepath = os.path.join(this_folder, f"{this_filename}_{varname}.{extension}")
            plt.savefig(this_filepath, dpi=300, facecolor="none")
        plt.show()
    plt.rcdefaults()
