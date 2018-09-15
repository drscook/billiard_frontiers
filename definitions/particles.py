class Particles():
    def __init__(self, dim=dim, num=part_num, mass=part_mass, radius=part_radius, gamma=part_gamma, temp=part_temp):
        self.dim = dim
        self.num = num
        self.mass = np.full(self.num, mass, dtype=np_dtype)
        self.radius = np.full(self.num, radius, dtype=np_dtype)

        g = np.sqrt(2/(2+self.dim))   # uniform mass distribution
        if gamma == 'shell':
            g = np.sqrt(2/self.dim)
        elif gamma == 'point':
            g = 0.0
        else:
            try:
                if (gamma >= 0) & (gamma <= np.sqrt(2/self.dim)):
                    g = gamma
            except:
                pass
        self.gamma = np.full(self.num, g, dtype=np_dtype)
        self.temp = np.full(self.num, temp, dtype=np_dtype)
        
        self.pos = np.full([self.num, self.dim], np.inf, dtype=np_dtype)
        self.pos_loc = self.pos.copy()
        self.vel = np.full([self.num, self.dim], np.inf, dtype=np_dtype)
        
        self.dim_spin = int(self.dim * (self.dim - 1) / 2)
        self.orient = np.full([self.num, self.dim, self.dim], np.inf, dtype=np_dtype)
        self.spin = np.full([self.num, self.dim, self.dim], np.inf, dtype=np_dtype)

        self.t = 0.0
        self.col = {}
        
        self.pp_collision_law = PP_SpecularLaw()
        
        self.t_hist = []
        self.pos_hist = []
        self.vel_hist = []
#         self.orient_hist = []
        self.spin_hist = []
        self.col_hist = []
        self.KE_lin_hist = []
        self.KE_ang_hist = []
        self.KE_tot_hist = []

    def get_mesh(self):
        S = sphere_mesh(self.dim, 1.0)
        if self.dim == 2:
            S = np.vstack([S, [-1,0]])

        part.mesh = []
        for p in range(part.num):
            self.mesh.append(S*self.radius[p]) # see visualize.py
        part.mesh = np.asarray(part.mesh)
            
        # pretty color for visualization
        cm = plt.cm.gist_rainbow
        idx = np.linspace(0, cm.N-1 , self.num).round().astype(int)
        self.clr = [cm(i) for i in idx]
        
    def get_pp_col_coefs(self, gap_only=False):
        dx = cross_subtract(part.pos)
        c0 = np.einsum('pqd, pqd -> pq', dx, dx)
        if gap_only == True:
            return np.sqrt(c0) - self.pp_gap_min
        c0 -= self.pp_gap_min**2
        dv = cross_subtract(self.vel)
        c1 = 2*np.einsum('pqd, pqd -> pq', dv, dx)
        c2 =   np.einsum('pqd, pqd -> pq', dv, dv)
        self.pp_col_coefs = np.array([c2, c1, c0]).T
        return self.pp_col_coefs

    def get_pp_gap(self):
        self.pp_gap = self.get_pp_col_coefs(gap_only=True)
        return self.pp_gap

    def get_pw_gap(self):
        self.pw_gap = np.array([w.get_pw_gap() for w in walls]).T
        return self.pw_gap

    def check_pos(self):
        self.get_pw_gap()
        pw_overlap = np.any(self.pw_gap < -thresh)
        if isinstance(part.pp_collision_law, PP_IgnoreLaw) == False:
            self.get_pp_gap()
            pp_overlap = np.any(self.pp_gap < -thresh)
        else:
            pp_overlap = False
        ok = not (pw_overlap or pp_overlap)
        return ok, not pp_overlap, not pw_overlap

    def rand_pos(self, p):
#         print('randomizing pos {}'.format(p))
        max_attempts = 1000
        for k in range(max_attempts):
            for d in range(self.dim):
                self.pos[p,d] = rng.uniform(-cell_size[d], cell_size[d])
            self.pos_loc[p] = self.pos[p].copy()
            if self.check_pos()[0] == True:
#                 print('Placed particle {}'.format(p))
                return 
        print(self.check_pos())
        raise Exception('Could not place particle {}'.format(p))

    def rand_vel(self, p):
        print('randomizing vel {}'.format(p))
        self.vel[p] = rng.normal(0.0, self.sigma_lin[p], size=dim)
    
    def rand_spin(self, p):
        v = [rng.normal(0.0, self.sigma_spin[p]) for d in range(self.dim_spin)]
        self.spin[p] = spin_mat_from_vec(v)
           
    def resolve_pp_collision(self, p1, p2):
        self.pp_collision_law.resolve_collision(self, p1, p2)

    def get_KE(self):
        self.KE_lin = self.mass * contract(self.vel**2) / 2
        self.KE_ang = self.mom_inert * contract(self.spin**2) / 4
        self.KE_tot = self.KE_lin + self.KE_ang
        return np.sum(self.KE_tot)

    def check_angular(self):
#         o_det = np.abs(np.linalg.det(self.orient))-1
#         orient_check = np.abs(o_det) < thresh
        skew = self.spin + np.swapaxes(self.spin, -2, -1)
        spin_check = contract(skew*skew) < thresh
        return np.all(spin_check) #and np.all(orient_check)
    
    def check(self):
        if self.check_pos()[0] == False:
            raise Exception('A particle escaped')
#         if abs(1-self.KE_init/self.get_KE()) > rel_tol:
#             raise Exception(' KE not conserved')
        if part.check_angular() == False:
            raise Exception('A particle has invalid orintation or spin matrix')
        return True
    
    def record_state(self):
        self.t_hist.append(self.t)
        self.pos_hist.append(self.pos.copy())
        self.vel_hist.append(self.vel.copy())
#         self.orient_hist.append(self.orient.copy())
        self.spin_hist.append(self.spin.copy())
        self.col_hist.append(self.col)
        self.KE_lin_hist.append(self.KE_lin)
        self.KE_ang_hist.append(self.KE_ang)
        self.KE_tot_hist.append(self.KE_tot)

    def clean_up(self):
        part.t_hist = np.asarray(part.t_hist)
        part.pos_hist = np.asarray(part.pos_hist)
        part.vel_hist = np.asarray(part.vel_hist)
#         part.orient_hist = np.asarray(part.orient_hist)
        part.spin_hist = np.asarray(part.spin_hist)
        part.KE_lin_hist = np.asarray(part.KE_lin_hist)
        part.KE_ang_hist = np.asarray(part.KE_ang_hist)
        part.KE_tot_hist = np.asarray(part.KE_tot_hist)
        part.num_steps = len(part.t_hist)
        part.num_frames = part.num_steps
        
#     def record(self):
#         rec = {'t_hist' : self.t_hist
#               ,'pos_hist' : self.pos_hist 
#               ,'vel_hist' : self.vel_hist 
#               ,'spin_hist' : self.spin_hist 
#               ,'radius' : self.radius
#               ,'pos_hist' : self.pos_hist 