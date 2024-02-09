import matplotlib.pyplot as plt
import numpy as np

from numpy import log
from scipy.optimize import curve_fit

class FitFn:
    def __init__(self, func, params_labels, func_str = None):
        self.fn = func
        self.labels = params_labels
        if not func_str is None:
            self.fn_label = func_str
        else:
            self.fn_label = "custom function"

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

lin_log = FitFn(
    func          = lambda x, a, b, c: a*x + b*log(x) + c,
    params_labels = ("a", "b", "c"),
    func_str      = "a*x + b*log(x) + c"
)

inv_log = FitFn(
    func          = lambda x, a, b, c: a/x + b*log(x) + c,
    params_labels = ("a", "b", "c"),
    func_str      = "a/x + b*log(x) + c"
)

lin_inv_log = FitFn(
    func          = lambda x, a, b, c, d: a*x + b/x + c*log(x) + d,
    params_labels = ("a", "b", "c", "d"),
    func_str      = "a*x + b/x + c*log(x) + d"
)

super_lin = FitFn(
    func          = lambda x, a, b, c, d: a*x*log(x) + b*x + c*log(x) + d,
    params_labels = ("a", "b", "c", "d"),
    func_str      = "a*x*log(x) + b*x + c*log(x) + d"
)


def print_fit(params, msg: str, fit_fn: FitFn):
    """Print fitting parameters with message `msg`"""
    print(f"\n{msg}")
    print(f"  {fit_fn.fn_label}")
    for name, param in zip(fit_fn.labels, params):
        print(f"    {name} = {param}")
    print()


def do_fit(xdata, ydata, fit_fn: FitFn = None, msg: str = None):
    """
    Fit data to the function
        a*x + b*log(x) + c
    Returns the fit params and the fitted function
    """
    if fit_fn is None:
        fit_fn = lin_log
    fit_params, _ = curve_fit(fit_fn.fn, xdata, ydata)
    fitted_fn = np.array([ fit_fn(x, *fit_params) for x in xdata])
    print_fit(fit_params, msg=msg, fit_fn=fit_fn)
    return fit_params, fitted_fn


def plot(xdata, ydata, tag: str, fit_fn: FitFn = None, label: str = None, msg: str = None):
    """Plot the fit"""
    if msg is None:
        msg = f"Fitting {tag}"
    if label is None:
        label = f"fit {tag}"
    params, fn = do_fit(xdata, ydata, fit_fn=fit_fn, msg=msg)
    plt.plot(xdata, fn, "--", label=label)


def with_fit(tag, fit_fn: FitFn=None, label=None, msg=None):
    """Decorator for adding fit capabilities"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            do_fit = kwargs.pop("do_fit", True)
            xdata, ydata = func(*args, **kwargs)
            if do_fit:
                plot(xdata, ydata, tag=tag, fit_fn=fit_fn, label=label, msg=msg)
                plt.legend()
        return wrapper
    return decorator


