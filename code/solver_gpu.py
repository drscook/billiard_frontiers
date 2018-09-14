@cuda.jit(device=True)
def solver_gpu(c2, c1, c0, mask):
    t0, t1 = solve_quadratic_gpu(c2, c1, c0)

    if mask == True:
        if abs(t0) < abs(t1):
            t0 = np.inf
        else:
            t1 = np.inf
    if t0 < 0:
        t0 = np.inf
        
    if t1 < 0:
        t1 = np.inf

    if t0 < t1:
        return t0
    else:
        return t1

    
@cuda.jit(device=True)
def solve_quadratic_gpu(c2, c1, c0):
    if abs(c2) <= abs(c0) * thresh:
        return solve_linear_gpu(c1, c0)
    else:
        d = c1**2 - 4*c0*c2
        if d < 0:
            return np.inf, np.inf
        else:
            if c1 < 0:
                s = -1
            else:
                s = 1
            d = -(c1 + s * math.sqrt(d)) / 2
            t_quad = d/c2   #quadratic formula
            t_cit = c0/d  #citardauq formula
            return t_quad, t_cit 

        
@cuda.jit(device=True)
def solve_linear_gpu(c1, c0):
    if abs(c1) <= thresh:
        return np.inf, np.inf
    else:
        return -c0 / c1, np.inf


@cuda.jit(device=True)
def row_min_gpu(A):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    m = float(cuda.blockDim.x)
    cuda.syncthreads()
    while m > 1:
        n = m / 2
        k = int(math.ceil(n))
        if (ty + k) < m:
            if A[tx,ty] > A[tx,ty+k]:
                A[tx,ty] = A[tx,ty+k]
        m = n
        cuda.syncthreads()