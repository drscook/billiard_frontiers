class Particles():
    def __init__(self, cell_size, num=1, max_steps=100, mass=1.0, radius=1.0, gamma='uniform', temp=1.0, pp_collision_law=PP_SpecularLaw, force=None, mode='serial'):
        self.dim = walls[0].dim
        self.max_steps = max_steps
        self.num = num
        self.mass = np.full(self.num, mass, dtype=np_dtype)        
        self.radius = np.full(shape=self.num, fill_value=radius, dtype=np_dtype)
        self.temp = np.full(self.num, temp, dtype=np_dtype)
        self.pp_collision_law = pp_collision_law()
        self.cell_size = np.asarray(cell_size, dtype = np_dtype)
        self.mode = mode
        self.force = force   # Currently, force only works for cylinders and must be axial.  We plan to generalize this in the future.
        part.terminate = False
        

        
        g = np.sqrt(2/(2+self.dim))   # uniform mass distribution
        if gamma == 'shell':
            g = np.sqrt(2/self.dim)
        elif gamma == 'point':
            g = 0.0
        else:
            try:
                if (gamma >= 0) & (gamma <= np.sqrt(2/self.dim)):
                    g = gamma
                    gamma = 'other'
            except:
                pass
        self.mass_dist = gamma
        self.gamma = np.full(self.num, g, dtype=np_dtype)
        
        self.pos = np.full([self.num, self.dim], np.inf, dtype=np_dtype)
        self.vel = np.full([self.num, self.dim], np.inf, dtype=np_dtype)
        self.spin = np.full([self.num, self.dim, self.dim], np.inf, dtype=np_dtype)

        self.t = 0.0
        self.col = {}
        
        self.record_params = ['mesh', 'dim', 'num', 'mass', 'radius', 'temp', 'mass_dist', 'gamma', 'mom_inert'
                            , 'run_path', 'data_filename', 'run_date', 'run_time', 'clr', 'cell_size', 'force'
                            , 'mode']
        self.record_vars = ['t', 'pos', 'spin']#, 'vel']


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
        cs = 2 * self.cell_size
        cell_idx = (self.pos[p] / cs).round()
        cell_idx[np.isinf(cell_idx)] = 0
        cell_offset = cell_idx * cs
        for k in range(max_attempts):
            for d in range(self.dim):
                self.pos_loc[p,d] = rng.uniform(-self.cell_size[d], self.cell_size[d])
            self.pos[p] = self.pos_loc[p] + cell_offset
            if self.check_pos()[0] == True:
#                 print('Placed particle {}'.format(p))
                return 
        raise Exception('Could not place particle {}'.format(p))


    def rand_vel(self, p):
#         print('randomizing vel {}'.format(p))
        self.vel[p] = rng.normal(0.0, self.sigma_vel[p], size=self.dim)


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
        spin_tranpose = np.swapaxes(self.spin, -2, -1) # transposes spin matrix of each particle
        skew = self.spin + spin_tranpose 
        spin_check = contract(skew**2) < thresh
        return np.all(spin_check)


    def check(self):
        if self.check_pos()[0] == False:
            raise Exception('A particle escaped')
        if self.check_angular() == False:
            raise Exception('A particle has an invalid spin matrix')
        return True

    
    ### Code below handles file i/o
    
    def get_params(self):
        def f(x):
            try:
                return x.tolist()
            except:
                return x
        d = {param:f(getattr(self, param)) for param in self.record_params}
        d['pp_collision_law'] = self.pp_collision_law.name            
        return d
        
    def record_init(self, free_mem_to_use):
        if self.write_to_file:
            date = str(datetime.date.today())            
            self.run_date = date
            self.run_time = time_stamp()

            date_path = root_path + date + '/'
            M = get_last_file_in_dir(date_path)            
            self.run_path = date_path + str(M+1) + '/'            
            self.data_filename = self.run_path + 'data.hdf5'
            self.part_params_filename = self.run_path + 'part_params.json'
            self.wall_params_filename = self.run_path + 'wall_params.json'
            
            os.makedirs(self.run_path, exist_ok=True)
            with open(self.part_params_filename, "w") as part_file:
                d = self.get_params()
                json.dump(d, part_file)

            with open(self.wall_params_filename, "w") as wall_file:
                d = [w.get_params() for w in walls]
                json.dump(d, wall_file)
            
            self.data_file = tables.open_file(self.data_filename, mode='w')
            print(f"I will write data to file {self.data_filename} at the period stated below.")

        ## Compute record length to keep in memory = file write period ##
        state_bytes = 0
        for v in self.record_vars:
            x = np.asarray(getattr(self, v)).astype(np_dtype)
            state_bytes += x.nbytes

        free_mem = psutil.virtual_memory().free
        avail_mem = free_mem * free_mem_to_use
        L = int(np.floor(avail_mem / state_bytes))
        self.hist_length = min(L, self.max_steps)
        print(f"I will keep the most recent {self.hist_length} steps in memory.")
    
        for v in self.record_vars:
            x = np.asarray(getattr(self, v)).astype(np_dtype)
            chunk = np.repeat(x[np.newaxis], self.hist_length, axis=0)
            setattr(self, f"{v}_hist", chunk)
            if self.write_to_file:
                tbl = self.data_file.create_earray(self.data_file.root, v
                                             ,tables.Atom.from_dtype(record_dtype(1).dtype)
                                             ,shape=(0, *x.shape)
                                             ,filters=table_filters
                                             ,chunkshape=chunk.shape
                                             ,expectedrows=self.max_steps)
    ## rec_dtype set in constants

    def record(self):
        period_complete = (self.record_ptr+1 >= self.hist_length)
        
        for v in self.record_vars:
            val = getattr(self, v)
            hist = getattr(self, f"{v}_hist")
            hist[self.record_ptr] = val
            if self.write_to_file & period_complete:
                self.data_file.root[v].append(hist.astype(record_dtype))
                
        if period_complete:            
            self.record_ptr = 0
        else:
            self.record_ptr += 1