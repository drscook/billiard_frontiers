def spin_mat_from_vec(v):
    # Converts spin vector to spin matrix
    # https://en.wikipedia.org/wiki/Rotation_matrix#Exponential_map
                     
    l = len(v)
    # l = d(d-1) -> d**2 - d - 2l = 0
    d = (1 + np.sqrt(1 + 8*l)) / 2
    if d % 1 != 0:
        raise Exception('vector {} of length {} converts to dim = {:.2f}.  Not integer.'.format(v,l,d))
    d = int(d)
    M = np.zeros([d,d])
    idx = np.triu_indices_from(M,1)
    s = (-1)**(np.arange(len(v))+1)
    w = v * s
    w = w[::-1]
    M[idx] = w
    M = make_symmetric(M, skew=True)
    return M

def spin_vec_from_mat(M):
    idx = np.triu_indices_from(M,1)
    w = M[idx]
    s = (-1)**(np.arange(len(w))+1)
    w = w[::-1]    
    v = w * s
    return v
   
def Pi_nu(v, nu):
    return v.dot(nu) * nu

def Pi(v, nu):
    w = Pi_nu(v ,nu)
    return v - w

def Lambda_nu(U, nu):
    return wedge(nu, U.dot(nu))

def E_nu(v, nu):
    return wedge(nu, v)

def Gamma_nu(U, nu):
    return U.dot(nu)