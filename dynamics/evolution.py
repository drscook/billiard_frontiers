def run_experiment(part, walls, record_period=1000, write_to_file=True):
    print()
    start = timer()
    
    with np.errstate(invalid='ignore'):
        initialize(part, walls)

    
    assert (record_period >= 1), 'record_period must be >= 1'
    part.record_period = min(record_period, max_steps)
    part.write_to_file = write_to_file
    
    part.record_ptr = 0
    part.record_init()
    part.record()
    
    for step in range(1,max_steps+1):
        next_state(part, walls)
        
        if part.mode == 'parallel':
            update_gpu(part)
        
        if part.record_ptr == 0:
            elapsed = timer() - start
            print(f"mode = {part.mode}, num_part = {part.num}, step = {step}, elapsed Time = {time_format(elapsed)}")
        part.record()


    elapsed = timer() - start
    print(f"mode = {part.mode}, num_part = {part.num}, step = {step}, elapsed Time = {time_format(elapsed)}")
    part.data_file.close()

    

def get_pw_dt_cpu(part, walls):
    part.pw_dt_cpu = np.array([solver(w.get_pw_col_coefs(), part.pw_mask[:, w.idx])[0] for w in walls]).T
    return part.pw_dt_cpu


def get_pp_dt_cpu(part, walls):
    part.pp_dt_cpu_full = solver(part.get_pp_col_coefs(), part.pp_mask)[0]
    part.pp_dt_cpu = np.min(part.pp_dt_cpu_full, axis=-1)
    return part.pp_dt_cpu

            
            
            
def initialize(part, walls):
    global get_dt
    if np.all([w.dim == part.dim for w in walls]) == False:
        raise Exception('Not all walls and part dimensions agree')
        
    if np.all((part.gamma >= 0) & (part.gamma <= np.sqrt(2/part.dim))) == False:
        raise Exception(f"illegal mass distribution parameter {gamma}")
        
    part.pw_gap_min = []
    for (i, w) in enumerate(walls):
        w.idx = i
        w.pw_gap_min = w.pw_gap_m * part.radius + w.pw_gap_b
        part.pw_gap_min.append(w.pw_gap_min)
        if isinstance(w.pw_collision_law, PW_IgnoreLaw):
            w.data[0,0] = -1
    part.pw_gap_min = np.asarray(part.pw_gap_min).T
    part.pp_gap_min = cross_subtract(part.radius, -part.radius)
    np.fill_diagonal(part.pp_gap_min, -1)
    
    part.mom_inert = part.mass * (part.gamma * part.radius)**2
    part.sigma_vel = np.sqrt(BOLTZ * part.temp / part.mass)
    part.sigma_spin = np.sqrt(BOLTZ * part.temp / part.mom_inert)

    part.pos_loc = part.pos.copy()
    for p in range(part.num):
        if np.any(np.isinf(part.pos[p])):
            part.rand_pos(p)
            
        if np.any(np.isinf(part.vel[p])):
            part.rand_vel(p)
            
        if np.any(np.isinf(part.spin[p])):
            part.spin[p,:,:] = 0.0
#             part.rand_spin(p)

    
    if same_initial_speeds:
        speed = np.linalg.norm(part.vel[0])
        part.vel = make_unit(part.vel) * speed

    part.pp_mask = np.full([part.num, part.num], False, dtype=bool)
    part.pw_mask = np.full([part.num, len(walls)], False, dtype=bool)
    
    part.get_mesh()
    for w in walls:
        w.get_mesh()
    part.KE_init = part.get_KE()
    part.check()

    
    part.pp_dt = np.array([np.inf])
    part.pw_dt = np.array([np.inf])
    if part.mode == 'serial':
        def get_dt(part, walls):
            part.pw_dt = get_pw_dt_cpu(part, walls)
            if (part.num > 1) & (not isinstance(part.pp_collision_law, PP_IgnoreLaw)):
                part.pp_dt = get_pp_dt_cpu(part, walls)

    elif part.mode == 'parallel':
        define_solver_gpu()
        init_gpu(part, walls)
        def get_dt(part, walls):
            part.pw_dt = get_pw_dt_gpu(part, walls)
            if check_gpu_cpu:
                get_pw_dt_cpu(part, walls)
                pw_check = np.allclose(part.pw_dt, part.pw_dt_cpu)
                if not pw_check:
                    raise Exception(f"cpu and gpu do not agree on pw_dt")

            if (part.num > 1) & (not isinstance(part.pp_collision_law, PP_IgnoreLaw)):
                part.pp_dt = get_pp_dt_gpu(part, walls)
                if check_gpu_cpu:
                    get_pp_dt_cpu(part, walls)
                    pp_check = np.allclose(part.pp_dt, part.pp_dt_cpu)
                    if not pp_check:
                        raise Exception(f"cpu and gpu do not agree on pp_dt")

    else:
        raise Exception(f"illegal mode {part.mode}")   
    
    

def next_state(part, walls, force=0):
    get_dt(part, walls)
    part.dt = min(np.min(part.pw_dt), np.min(part.pp_dt))
    part.pw_mask[:] = False
    part.pp_mask[:] = False

    if np.isinf(part.dt):
        raise Exception("No future collisions detected")
        
    if part.force is None:
        part.pos += part.vel * part.dt
        part.pos_loc += part.vel * part.dt
    else:  # Currently, force only works for cylinders and must be axial.  We plan to generalize this in the future.
        part.pos[:,-1] += force * part.dt**2 /(2 * part.mass) + part.vel[:,-1] * part.dt
        part.pos_loc[:,-1] += force * part.dt**2 /(2 * part.mass) + part.vel[:,-1] * part.dt
        
        part.pos[:,0:-1] += part.vel[:,0:-1] * part.dt
        part.pos_loc[:,0:-1] += part.vel[:,0:-1] * part.dt
        part.vel[:,-1] += force/part.mass * part.dt

    part.t += part.dt
        
    pw_events = (part.pw_dt - part.dt) < thresh
    pw_counts = np.sum(pw_events, axis=-1)
    pw_tot = np.sum(pw_events)
    
    pp_events = (part.pp_dt - part.dt) < thresh
    pp_counts = pp_events
    pp_tot = np.sum(pp_events)
    
    if (pw_tot == 0) & (pp_tot == 2):
        p, q = np.nonzero(pp_counts)[0]
        part.col = {'p':p, 'q':q}
        part.pp_mask[p,q] = True
        part.pp_mask[q,p] = True
        part.resolve_pp_collision(p, q)
    elif (pw_tot == 1) & (pp_tot == 0):
        p = np.argmax(pw_counts)
        w = np.argmax(pw_events[p])
        part.col = {'p':p, 'w':w}
        part.pw_mask[p,w] = True
        walls[w].resolve_pw_collision(part, walls, p)
    else:
        P = np.nonzero(pp_counts + pw_counts)[0]
        print('COMPLEX COLLISION DETECTED. Re-randomizing positions of particles {}'.format(P))
        part.record()  # record state before and after re-randomizing position to animations look right
        for p in P:
            part.rand_pos(p)
    part.check()