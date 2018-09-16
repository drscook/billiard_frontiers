class Particles():
    def __init__(self, dim=dim, num=part_num, mass=part_mass, radius=part_radius, gamma=part_gamma, temp=part_temp, pp_collision_law=PP_SpecularLaw):
        self.dim = dim
        self.num = num
        self.mass = np.full(self.num, mass, dtype=np_dtype)        
        self.radius = np.full(shape=self.num, fill_value=radius, dtype=np_dtype)
        self.temp = np.full(self.num, temp, dtype=np_dtype)
        self.pp_collision_law = pp_collision_law()
        
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
        
        self.pos = np.full([self.num, self.dim], np.inf, dtype=np_dtype)
        self.pos_loc = self.pos.copy()
        self.vel = np.full([self.num, self.dim], np.inf, dtype=np_dtype)
        self.spin = np.full([self.num, self.dim, self.dim], np.inf, dtype=np_dtype)

        self.t = 0.0
        self.col = {}
        
        self.record_params = ['mesh', 'dim', 'num', 'mass', 'radius', 'temp', 'gamma', 'mom_inert'
                            , 'run_path', 'data_filename', 'run_date', 'run_time', 'clr']
        self.record_vars = ['t', 'pos', 'vel', 'spin']



    def get_mesh(self):
        S = sphere_mesh(self.dim, 1.0)
        if self.dim == 2:
            S = np.vstack([S, [-1,0]])

        self.mesh = []
        for p in range(self.num):
            self.mesh.append(S*self.radius[p]) # see visualize.py
        self.mesh = np.asarray(self.mesh)
            
        # pretty color for visualization
        cm = plt.cm.gist_rainbow
        idx = np.linspace(0, cm.N-1 , self.num).round().astype(int)
        self.clr = [cm(i) for i in idx]
        
    def get_pp_col_coefs(self, gap_only=False):
        dx = cross_subtract(self.pos)
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
        if isinstance(self.pp_collision_law, PP_IgnoreLaw) == False:
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
#         print('randomizing vel {}'.format(p))
        self.vel[p] = rng.normal(0.0, self.sigma_vel[p], size=dim)
    
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
        if self.check_angular() == False:
            raise Exception('A particle has invalid orintation or spin matrix')
        return True
    
    def params(self):
        def f(x):
            try:
                return x.tolist()
            except:
                return x
        d = {param:f(getattr(self, param)) for param in self.record_params}
        d['pp_collision_law'] = self.pp_collision_law.name            
        return d
        
    def record_init(self):
        if self.write_to_file:
            date = str(datetime.date.today())            
            self.run_date = date
            self.run_time = time_stamp()

            date_path = root_path + date + '/'
            M = get_last_file_in_dir(date_path)            
            self.run_path = date_path + str(M+1) + '/'
            os.makedirs(self.run_path, exist_ok=True)
           
            self.data_filename = self.run_path + 'data.hdf5'
            self.data_file = tables.open_file(self.data_filename, mode='w')
            
            self.part_params_filename = self.run_path + 'part_params.json'
            with open(self.part_params_filename, "w") as part_file:
                d = self.params()
                json.dump(d, part_file)

            self.wall_params_filename = self.run_path + 'wall_params.json'
            with open(self.wall_params_filename, "w") as wall_file:
                d = [w.params() for w in walls]
                json.dump(d, wall_file)
            
        for v in self.record_vars:
            x = np.asarray(getattr(self, v))
            z = np.zeros(shape=(self.record_period, *x.shape), dtype=x.dtype)
            z[0] = x.copy()
            setattr(self, f"{v}_hist", z)
            if self.write_to_file:
                tbl = self.data_file.create_earray(self.data_file.root, v
                                             ,tables.Atom.from_dtype(x.dtype)
                                             ,shape=(0, *x.shape)                                   
                                             ,filters=table_filters
                                             ,chunkshape=z.shape
                                             ,expectedrows=max_steps)
#                 tbl.append(z)
                setattr(self, f"{v}_storage", tbl)                
    

    def record(self):
        period_complete = (self.record_ptr+1 >= self.record_period)
        
        for v in self.record_vars:
            val = getattr(self, v)
            hist = getattr(self, f"{v}_hist")
            hist[self.record_ptr] = val
            if self.write_to_file & period_complete:
                tbl = getattr(self, f"{v}_storage")
                tbl.append(hist)

        if period_complete:
            print('Writing to file')
            self.record_ptr = 0
        else:
            self.record_ptr += 1