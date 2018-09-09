from copy import copy
import math
import IPython.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

class Formatter():
    def __init__(self, digits=4, max_rows=20, max_cols=None):
        self.digits = digits
        self.max_rows = max_rows
        self.max_cols = max_cols

    def nbr_formatter(self, x):
        mag = math.floor(math.log10(x)) + 1
        if mag < -self.digits:
            s = f"{x:.{self.digits-1}f}"
        elif mag > self.digits:
            s = f"{x:.{self.digits-2}e}"
        else:
            before = max(mag, 1)
            after = self.digits - before
            s = f"{x:{before}.{after}f}"
        return s
                   
    def display(self, X):
        with pd.option_context('display.max_rows', self.max_rows
                              ,'display.max_columns', self.max_cols
                              ,'display.float_format', self.nbr_formatter):
            if isinstance(X, pd.Series) or (isinstance(X, np.ndarray) and X.ndim <=2):
                IPython.display.display(pd.DataFrame(X))
            else:
                IPython.display.display(X)

default_formatter = Formatter()

np.set_printoptions(formatter = {'float': default_formatter.nbr_formatter})
pd.options.display.float_format = default_formatter.nbr_formatter
pd.options.display.notebook_repr_html = True
pd.options.display.show_dimensions = True

def display(X, formatter=None, digits=None, max_rows=None, max_cols=None):
    if not formatter: formatter = copy(default_formatter)
    formatter = copy(formatter)
    if digits: formatter.digits = digits
    if max_rows: formatter.max_rows = max_rows
    if max_cols: formatter.max_cols = max_cols
    formatter.display(X)