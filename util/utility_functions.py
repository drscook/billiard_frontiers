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
        
        
#########################################################################################################################
### LINEAR ALGEBRA ###
#########################################################################################################################
        
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