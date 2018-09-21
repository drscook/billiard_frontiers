##################################################################
### Common Imports ###
##################################################################

import os
from copy import copy, deepcopy
import math
import itertools as it
import IPython.display
from timeit import default_timer as timer
import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


try:
    import tables
except:
    os.system('pip install --upgrade tables')
    import tables

try:
    import numba as nb
except:
    os.system('pip install --upgrade numba')
    import numba as nb


    
##################################################################
### Display Preferences ###
### Everything in this section is optional and can be turned off
### if it causes errors.  It is just trying to print arrays
### with nice rounding and other options.  But it has not been
### thoroughly tested.
##################################################################

plt.style.use("fivethirtyeight")

class Formatter():
    def __init__(self, digits=4, max_rows=20, max_cols=None):
        self.digits = digits
        self.max_rows = max_rows
        self.max_cols = max_cols

    def nbr_formatter(self, x):
        try:
            mag = math.floor(math.log10(x)) + 1
            if mag < -self.digits:
                s = f"{x:.{self.digits-1}f}"
            elif mag > self.digits:
                s = f"{x:.{self.digits-2}e}"
            else:
                before = max(mag, 1)
                after = self.digits - before
                s = f"{x:{before}.{after}f}"
        except:
            s = str(x)
        return s
                   
    def display(self, X):
        with pd.option_context('display.max_rows', self.max_rows
                              ,'display.max_columns', self.max_cols
                              ,'display.float_format', self.nbr_formatter):
            try:
                IPython.display.display(pd.DataFrame(X))
            except:
                IPython.display.display(X)

default_formatter = Formatter()

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

    
    
##################################################################
### Utility Functions ###
##################################################################


def insert_totals(df):
    df = pd.DataFrame(df)
    col_sums = df.sum(axis=0)
    df.loc['TOTAL'] = col_sums
    row_sums = df.sum(axis=1)
    df['TOTAL'] = row_sums
    return df


def time_format(x):
    h, m = np.divmod(x, 3600)
    m, s = np.divmod(m, 60)
    return f"{int(h):02d} hr {int(m):02d} min {s:05.02f} sec"


def time_stamp():
    now = datetime.datetime.now()
    return f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{round(now.microsecond/10000)}"


def get_last_file_in_dir(path):
        os.makedirs(path, exist_ok=True)
        ls = os.listdir(path)
        ls.append(-1)
        def f(x):
            try:
                return int(x)
            except:
                return -1
        return sorted(map(f,ls))[-1]


def listify(X):
    """
    Ensure X is a list
    """
    if isinstance(X, list):
        return X
    elif (X is None) or (X is np.nan):
        return []
    elif isinstance(X,str):
        return [X]
    else:
        try:
            return list(X)
        except:
            return [X]
        
        
        
##################################################################
### LINEAR ALGEBRA ###
##################################################################


def cross_subtract(u,v=None):
    """
    w[i,j] = u[i] - v[j] letting inf-inf=inf
    """
    if v is None:
        v=u.copy()
    with np.errstate(invalid='ignore'):  # suppresses warnings for inf-inf
        w = u[:,np.newaxis] - v[np.newaxis,:]
        w[np.isnan(w)] = np.inf
    return w


def wedge(a,b):
    """
    Geometric wedge product
    """
    return np.outer(b,a)-np.outer(a,b)


def contract(A, keepdims=[0]):
    """
    Sum all dimensions except those in keepdims
    """
    keepdims = listify(keepdims)
    A = np.asarray(A)
    return np.einsum(A, range(A.ndim), keepdims)


def make_unit(A, axis=-1):
    """
    Normalizes along given axis so the sum of squares is 1
    """
    A = np.asarray(A)
    M = np.linalg.norm(A, axis=axis, keepdims=True)
    return A / M


def make_symmetric(A, skew=False):
    """
    Returns symmetric or skew-symmatric matrix by copying upper triangular onto lower.
    """
    A = np.asarray(A)
    U = np.triu(A,1)
    if skew == True:
        return U - U.T
    else:
        return np.triu(A,0) + U.T    
    
    
    
    
##################################################################
### Random ###
##################################################################

    
def random_uniform_sphere(num=1, dim=2, radius=1.0):
    pos = rng.normal(size=[num, dim])
    pos = make_unit(pos, axis=1)
    return abs(radius) * pos


def random_uniform_ball(num=1, dim=2, radius=1.0):
    pos = random_uniform_sphere(num, dim, radius)
    r = rng.uniform(size=[num, 1])
    return r**(1/dim) * pos