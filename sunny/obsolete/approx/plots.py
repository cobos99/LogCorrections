import numpy as np
import matplotlib.pyplot as plt

import approx.even_case as even
import approx.odd_case as odd
import approx.fit as fit
from approx.fit import with_fit


marker_options = dict(markersize=10, markeredgewidth=3)
#--------------------------------------------------
# Difference between odd and even cases
#--------------------------------------------------

# Even case
#
@with_fit("sum even full")
def sum_even_full(lengths):
    full_sum = np.array([even.full_sum(l) for l in lengths])
    plt.plot(lengths, full_sum, "1", label="sum even full", **marker_options)
    return lengths, full_sum

@with_fit("sum even simpl.")
def sum_even_simpl(lengths):
    simpl_sum = np.array([even.simpl_sum(l) for l in lengths])
    plt.plot(lengths, simpl_sum, "2", label="sum even simpl.", **marker_options)
    return lengths, simpl_sum

def sum_even(lengths, do_fit=True):
    sum_even_full(lengths, do_fit=do_fit)
    sum_even_simpl(lengths, do_fit=do_fit)
    plt.ylabel(r"$\Sigma^{\text{even}}$")
    plt.xlabel("$\ell$")
    plt.legend()


# Odd case
#
@with_fit("sum odd full")
def sum_odd_full(lengths):
    full_sum = np.array([odd.full_sum(l) for l in lengths])
    plt.plot(lengths, full_sum, "1", label="sum odd full", **marker_options)
    return lengths, full_sum

@with_fit("sum odd simpl.")
def sum_odd_simpl(lengths):
    simpl_sum = np.array([odd.simpl_sum(l) + odd.delta(l) for l in lengths])
    plt.plot(lengths, simpl_sum, "2", label="sum odd simpl.", **marker_options)
    return lengths, simpl_sum

def sum_odd(lengths, do_fit=True):
    sum_odd_full(lengths, do_fit=do_fit)
    sum_odd_simpl(lengths, do_fit=do_fit)
    plt.ylabel(r"$\Sigma^{\text{odd}}$")
    plt.xlabel("$\ell$")
    plt.legend()

# Both
#
def sums(lengths, do_fit=True, figsize=(10, 5)):
    plt.subplots(1, 2, figsize=figsize)
    plt.subplot(1, 2, 1)
    sum_even(lengths, do_fit=do_fit)
    plt.title("Sums for the even case")
    plt.subplot(1, 2, 2)
    plt.title("Sums for the odd case")
    sum_odd(lengths, do_fit=do_fit)

#--------------------------------------------------
# Plot the difference between the even and odd case
#--------------------------------------------------

@with_fit("diff. of sums (full)")
def sum_diff_full(lengths):
    diff = np.array([ odd.full_sum(l) - even.full_sum(l) for l in lengths])
    plt.plot(lengths, diff, "1", label="diff. of sums (full)", **marker_options)
    plt.xlabel("$\ell$")
    plt.ylabel(r"$\Sigma^{\text{odd}} - \Sigma^{\text{even}}$")
    plt.legend()
    return lengths, diff


@with_fit("diff. of sums (simpl.)")
def sum_diff_simpl(lengths, with_delta=True):
    if with_delta:
        diff = np.array([ odd.simpl_sum(l) + odd.delta(l) - even.simpl_sum(l) for l in lengths])
    else:
        diff = np.array([ odd.simpl_sum(l) - even.simpl_sum(l) for l in lengths])
    plt.plot(lengths, diff, "2", label="diff. of sums (simpl.)", **marker_options)
    plt.xlabel("$\ell$")
    plt.ylabel(r"$\Sigma^{\text{odd}} - \Sigma^{\text{even}}$")
    plt.legend()
    return lengths, diff


def sum_diff(lengths, do_fit=True, with_delta=False, figsize=(10, 5)):
    plt.subplots(1, 2, figsize=figsize)
    plt.subplot(1, 2, 1)
    sum_diff_full(lengths, do_fit=do_fit)
    plt.title("Sums difference (full)")
    plt.subplot(1, 2, 2)
    sum_diff_simpl(lengths, do_fit=do_fit, with_delta=with_delta)
    plt.title("Sums difference (simpl.)")


#--------------------------------------------------
# Plot the sums $\Phi$
#--------------------------------------------------

@with_fit("phi even")
def phi_even(lengths, do_fit=True):
    sum_even = np.array([even.phi(l) for l in lengths])
    plt.plot(lengths, sum_even, "1", label="even", **marker_options)
    plt.xlabel("$\ell$")
    plt.ylabel(r"$\Phi^{\text{even}}$")
    plt.legend()
    return lengths, sum_even


@with_fit("phi odd")
def phi_odd(lengths, do_fit=True):
    sum_odd  = np.array([odd.phi(l) for l in lengths])
    plt.plot(lengths, sum_odd, "x", label="odd", **marker_options)
    plt.xlabel("$\ell$")
    plt.ylabel(r"$\Phi^{\text{odd}}$")
    plt.legend()
    return lengths, sum_odd


def phi(lengths, do_fit=True, figsize=(10, 5)):
    plt.subplots(1, 2, figsize=figsize)
    plt.subplot(1, 2, 1)
    phi_even(lengths, do_fit=do_fit)
    plt.title(r"Sum $\Phi$ for the even case")
    plt.subplot(1, 2, 2)
    phi_odd(lengths, do_fit=do_fit)
    plt.title(r"Sum $\Phi$ for the odd case")
    plt.tight_layout()


@with_fit("phi diff", fit_fn=fit.inv_log)
def phi_diff(lengths):
    phi_diff = np.array([odd.phi(l) - even.phi(l) for l in lengths])
    plt.plot(lengths, phi_diff, "x", label="phi diff.", **marker_options)
    plt.xlabel("$\ell$")
    plt.ylabel(r"$\Phi^{\text{odd}} - \Phi^{\text{even}}$")
    return lengths, phi_diff


#--------------------------------------------------
# Plot the Delta term that appears for the odd case
#--------------------------------------------------

@with_fit("delta term")
def delta(lengths):
    deltas = np.array([odd.delta(l) for l in lengths])
    plt.plot(lengths, deltas, "x", label="computed", **marker_options)

    # plt.title("Deviation term for odd case")
    plt.xlabel("$\ell$")
    plt.ylabel(r"$\Delta$")
    return lengths, deltas


#--------------------------------------------------
# Plot the normalization constant
#--------------------------------------------------

@with_fit("norm even", fit_fn=fit.super_lin)
def norm_even(lengths):
    norms = [even.log_norm(l) for l in lengths]
    plt.plot(lengths, norms, "1", label="norm even", **marker_options)
    plt.xlabel("$\ell$")
    plt.ylabel(r"$\log\mathcal{N}^{\text{even}}$")
    plt.legend()
    return lengths, norms

@with_fit("norm even", fit_fn=fit.super_lin)
def norm_odd(lengths):
    norms = [odd.log_norm(l) for l in lengths]
    plt.plot(lengths, norms, "2", label="norm odd", **marker_options)
    plt.xlabel("$\ell$")
    plt.ylabel(r"$\log\mathcal{N}^{\text{odd}}$")
    plt.legend()
    return lengths, norms


def norm(lengths, do_fit=True):
    norm_even(lengths, do_fit=do_fit)
    norm_odd(lengths, do_fit=do_fit)
    plt.title("Log of the normalization constants")


@with_fit("norm diff")
def norm_diff(lengths):
    norm_diff = [odd.log_norm(l) - even.log_norm(l) for l in lengths]
    plt.plot(lengths, norm_diff, "x", label="norm diff", **marker_options)
    plt.title(r"Difference between the $\log\mathcal{N}$ terms")
    plt.xlabel("$\ell$")
    plt.ylabel(r"$\log\mathcal{N}^{\text{odd}} - \log\mathcal{N}^{\text{even}}$")
    plt.legend()
    return lengths, norm_diff


def full_diff(lengths):
    norm_diff = np.array([odd.log_norm(l) - even.log_norm(l) for l in lengths])
    sum_diff = np.array([ odd.full_sum(l) - even.full_sum(l) for l in lengths])
    plt.plot(lengths, -norm_diff + 2*sum_diff)

def standard_look():
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6)
    plt.title("")
    legend = plt.legend()
    legend.get_texts()[0].set_text('Numerical')
    legend.get_texts()[1].set_text('Fit')

def save(tag):
    plt.savefig(f"graphs/XX_{tag}.pdf", bbox_inches="tight")
