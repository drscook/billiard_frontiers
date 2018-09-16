def solver(coefs, mask):
    """
    Smart solvers that applies fast analytic linear and quadratic forumlae when /
    possible.  Defaults to slower np.roots base on eigenvalues of the companion matrix.
    """
    with np.errstate(invalid='ignore'):
        sh = list(coefs.shape)
        sh[-1] -= 1
        deg = sh[-1]

        if deg == 1:
    #         print('linear')
            c1, c0 = np.rollaxis(coefs, -1, 0)
            roots = solve_linear(c1, c0)[:,np.newaxis]

        elif deg == 2:
    #         print('quadratic')
            c2, c1, c0 = np.rollaxis(coefs, -1, 0)
            roots = solve_quadratic(c2, c1, c0)

        else:
    #         print('general')
            roots = solve_general(coefs)

        # applies mask to hide the 0 time associated to the prior collision event
        if np.sum(mask) > 0:
            mag = np.full(roots.shape, np.inf)
            mag[mask] = np.abs(roots[mask])
            mag_min = np.min(mag, axis=-1, keepdims=True)
            idx = (mag <= mag_min) & mask[...,np.newaxis]
            roots[idx] = np.inf

        # removes complex and negative roots
        roots[np.iscomplex(roots)] = np.inf
        roots = np.real(roots)
        roots[roots<0] = np.inf
        t = np.min(roots, axis=-1)
    return t, roots


def solve_quadratic(c2, c1, c0):
    sh = list(c0.shape)
    deg = 2
    sh.append(deg)
    roots = np.full(sh, np.inf)
    lin = np.abs(c2) <= np.abs(c0) * thresh
    quad = ~lin
    roots[lin, 0] = solve_linear(c1[lin], c0[lin])
    s = np.sign(c1)
    d = s.copy()
    d[quad] = c1[quad]**2 - 4*c0[quad]*c2[quad]
    real = quad & (d >= 0)    
    d[real] = -(c1[real] + s[real] * np.sqrt(d[real])) / 2
    roots[real, 0] = d[real] / c2[real] #quadratic formula
    roots[real, 1] = c0[real] / d[real] #citardauq formula
    return roots


def solve_linear(c1, c0):
    sh = list(c0.shape)
    roots = np.full(sh, np.inf)
    lin = np.abs(c1) > thresh
    roots[lin] = -1 * c0[lin] / c1[lin]
    return roots

def solve_general(coefs):
    sh = list(coefs.shape)
    sh[-1] -= 1
    deg = sh[-1]
    coefs = coefs.reshape(-1, deg+1).astype(complex)
    N = coefs.shape[0]
    roots = np.full([N, deg], np.inf).astype(complex)
    idx = ~np.any(np.isinf(coefs), axis=-1)
    roots[idx] = my_roots(coefs[idx], roots[idx])
    return roots.reshape(sh)

@nb.jit(nopython=True)
def my_roots(coefs, roots):
    for i in range(coefs.shape[0]):
        r = np.roots(coefs[i])
        roots[i, :len(r)] = r
    return roots