import billiards.util
import billiards.util.superimporter

import os
import itertools as it
import math
import IPython.display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from billiards.util import utility_functions as util
from timeit import default_timer as timer


display_digits = 3
def nbr_formatter(x):
    mag = math.floor(math.log10(x)) + 1
    if mag < -1*digits:
        s = f"{x:.{digits-1}f}"
    elif mag > digits:
        s = f"{x:.{digits-2}e}"
    else:
        before = max(mag, 1)
        after = digits - before
        s = f"{x:{before}.{after}f}"
    return s

np.set_printoptions(formatter = {'float':nbr_formatter})
pd.options.display.float_format = nbr_formatter
pd.options.display.notebook_repr_html = True
pd.options.display.show_dimensions = True
pd.options.display.max_rows = 20
pd.options.display.max_columns = None

def display(X):
    if isinstance(X, pd.Series) or (isinstance(X, np.ndarray) and X.ndim <=2):
        IPython.display.display(pd.DataFrame(X))
    else:
        IPython.display.display(X)
    return



plt.style.use("fivethirtyeight")