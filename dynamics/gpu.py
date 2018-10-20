import numba.cuda as cuda

def setup_gpu():
    """
    Installs drivers and gets specs.  Run before setting up billiards.
    """
    global threads_per_block_max, optimal_part_num, optimal_part_num, nb_dtype
    
    setup_numba_cuda()

    try:
        threads_per_block_max = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    except:
        threads_per_block_max = 2**10
        print(f"cuda.get_current_device().MAX_THREADS_PER_BLOCK failed.  Using default {threads_per_block_max}.")
        
    optimal_part_num = int(np.floor(np.sqrt(threads_per_block_max)))
    nb_dtype = nb.float64
    

def update_gpu(part):
    cuda.to_device(part.pos, to=part.pos_dev)
    cuda.to_device(part.pos_loc, to=part.pos_loc_dev)
    cuda.to_device(part.vel, to=part.vel_dev)
    cuda.to_device(part.pw_events_old, to=part.pw_events_dev)  
    cuda.to_device(part.pp_events_old, to=part.pp_events_dev)

    

def init_gpu(part, walls):
    global dim
    global pw_blk_rows, pw_blk_cols, pw_grd_rows, pw_grd_cols
    global pw_blk_shp, pw_grd_shp, pw_tot_shp
    global pp_blk_rows, pp_blk_cols, pp_grd_rows, pp_grd_cols
    global pp_blk_shp, pp_grd_shp, pp_tot_shp
    global get_pp_dt_gpu, get_pw_dt_gpu
    
    dim = part.dim

    bw = min(threads_per_block_max, part.num)
    pw_blk_rows = bw
    pw_blk_cols = 1

    gw = int(np.ceil(part.num / bw))
    pw_grd_rows = gw
    pw_grd_cols = len(walls)
    
    pw_blk_shp = (pw_blk_rows, pw_blk_cols)
    pw_grd_shp = (pw_grd_rows, pw_grd_cols)
    pw_tot_shp = (pw_grd_rows, pw_grd_cols, pw_blk_rows, pw_blk_cols)

    bp = min(optimal_part_num, part.num)
    pp_blk_rows = bp
    pp_blk_cols = bp

    gp = int(np.ceil(part.num / bp))
    pp_grd_rows = gp
    pp_grd_cols = gp

    pp_blk_shp = (pp_blk_rows, pp_blk_cols)
    pp_grd_shp = (pp_grd_rows, pp_grd_cols)
    pp_tot_shp = (pp_grd_rows, pp_grd_cols, pp_blk_rows, pp_blk_cols)

    part.walls_data = np.array([wall.data.copy() for wall in walls], dtype=np_dtype)
    part.walls_data_dev = cuda.to_device(part.walls_data)

    part.pw_gap_min_dev = cuda.to_device(part.pw_gap_min)
    part.pp_gap_min_dev = cuda.to_device(part.pp_gap_min)
    
    part.pos_dev = cuda.to_device(part.pos)
    part.pos_loc_dev = cuda.to_device(part.pos_loc)
    part.vel_dev = cuda.to_device(part.vel)
    
    ## We wish to minimize data transfer between gpu and cpu for the sake of speed.
    ## The want to avoid passing all p x p times because these are floats.
    ## Instead, we will pass only the smallest time from each block via pp_dt_blk_gpu
    ## and a BOOLEAN array called pp_events
    ## An entry of pp_events is True only if the dt for that pair of particles
    ## is within thresh of the min_dt on its block.
    ## Now, the true GLOBAL min_dt may be from a different block
    ## So, once passed to the cpu, we find the global min_dt = min(pp_dt_blk)
    ## and set all entries of pp_events to False except those from 
    ## blocks that achieve this global min_dt
    
    ## Repeat for particle-wall
    
    ## Finally, if the min_dt over p-p interactions is bigger than the min_dt over p-w,
    ## we set every entry of pp_events to False.  Conversely for pw_events.
    
    part.pw_events_dev = cuda.to_device(part.pw_events_old)
    part.pp_events_dev = cuda.to_device(part.pp_events_old)
    
    part.pw_dt_blk_gpu = cuda.pinned_array(pw_grd_shp, dtype=np_dtype)
    part.pw_dt_blk_gpu[:] = np.inf
    part.pw_dt_blk_dev = cuda.to_device(part.pw_dt_blk_gpu)
    
    part.pp_dt_blk_gpu = cuda.pinned_array(pp_grd_shp, dtype=np_dtype)
    part.pp_dt_blk_gpu[:] = np.inf
    part.pp_dt_blk_dev = cuda.to_device(part.pp_dt_blk_gpu)

    # Though inefficient, we may occasionally want to pass all pxp and pxw dt's
    # back to CPU for validation and debugging.  So, we create pp_dt_gpu and pw_dt_gpu
    # for this purpose.  In other cases, it is set as a 1x1 matrix and ignored
    # (other than as a placeholder in function signatures and calls).
    
    if check_gpu_cpu:
        pw_sh = [part.num, len(walls)]
        pp_sh = [part.num, part.num]
    else:
        pw_sh = [1, 1]
        pp_sh = [1, 1]
        
    part.pw_dt_gpu = cuda.pinned_array(pw_sh, dtype=np_dtype)
    part.pw_dt_gpu[:] = np.inf
    part.pw_dt_dev = cuda.to_device(part.pw_dt_gpu)

    part.pp_dt_gpu = cuda.pinned_array(pp_sh, dtype=np_dtype)
    part.pp_dt_gpu[:] = np.inf
    part.pp_dt_dev = cuda.to_device(part.pp_dt_gpu)

    update_gpu(part)


    @cuda.jit(device=False)
    def get_pp_dt_kernel(pos, vel, pp_gap_min, pp_events_dev, pp_dt_blk_dev, pp_dt_dev):
        row_loc = cuda.threadIdx.x
        col_loc = cuda.threadIdx.y
        idx_loc = row_loc * cuda.blockDim.y + col_loc

        blk_row = cuda.blockIdx.x
        blk_col = cuda.blockIdx.y
        row_glob = blk_row * cuda.blockDim.x + row_loc 
        col_glob = blk_col * cuda.blockDim.y + col_loc
        idx_glob = row_glob * cuda.blockDim.y * cuda.gridDim.y + col_glob
        
        p = row_glob
        q = col_glob
        N = pos.shape[0]

        pos1_shr  = cuda.shared.array(shape=(pp_blk_rows, dim), dtype=nb_dtype)
        pos2_shr  = cuda.shared.array(shape=(pp_blk_cols, dim), dtype=nb_dtype)
        vel1_shr  = cuda.shared.array(shape=(pp_blk_rows, dim), dtype=nb_dtype)
        vel2_shr  = cuda.shared.array(shape=(pp_blk_cols, dim), dtype=nb_dtype)
        pp_dt_shr = cuda.shared.array(shape=(pp_blk_rows, pp_blk_cols), dtype=nb_dtype)
        temp_shr  = cuda.shared.array(shape=(pp_blk_rows, pp_blk_cols), dtype=nb_dtype)
        
        d = col_loc
        if d < dim:
            pos1_shr[row_loc, d] = pos[p, d]
            vel1_shr[row_loc, d] = vel[p, d]
        
        d = row_loc
        if d < dim:
            pos2_shr[col_loc, d] = pos[q, d]
            vel2_shr[col_loc, d] = vel[q, d]

        cuda.syncthreads()

        if (p < N) and (q < N):
            c0 = 0.0
            c1 = 0.0
            c2 = 0.0
            for d in range(dim):
                dx = pos1_shr[row_loc, d] - pos2_shr[col_loc, d]
                dv = vel1_shr[row_loc, d] - vel2_shr[col_loc, d]
                c0 += (dx**2)
                c1 += (dx*dv*2)
                c2 += (dv**2)
            c0 -= (pp_gap_min[p,q]**2)
            dt = solver_gpu(c2, c1, c0, pp_events_dev[p, q])
        else:
            dt = np.inf

        pp_dt_shr[row_loc, col_loc] = dt
        temp_shr[row_loc, col_loc] = dt
        cuda.syncthreads()
        min_gpu(temp_shr)
        min_dt = temp_shr[0, 0]

        if (p < N) and (q < N):
            pp_events_dev[p, q] = (dt < min_dt + thresh)
            if idx_loc == 0:
                pp_dt_blk_dev[blk_row, blk_col] = min_dt
            if check_gpu_cpu:
                pp_dt_dev[p, q] = dt
            
        

    @cuda.jit(device=False)
    def get_pw_dt_kernel(pos_loc, vel, pw_gap_min, walls, pw_events_dev, pw_dt_blk_dev, pw_dt_dev):
        row_loc = cuda.threadIdx.x
        col_loc = cuda.threadIdx.y
        idx_loc = row_loc * cuda.blockDim.y + col_loc

        blk_row = cuda.blockIdx.x
        blk_col = cuda.blockIdx.y
        row_glob = blk_row * cuda.blockDim.x + row_loc 
        col_glob = blk_col * cuda.blockDim.y + col_loc
        idx_glob = row_glob * cuda.blockDim.y * cuda.gridDim.y + col_glob
        
        p = row_glob
        w = col_glob
        N = pos_loc.shape[0]
        W = walls.shape[0]
        shape = walls[w,0,0]
        
        pw_dt_shr = cuda.shared.array(shape=(pw_blk_rows, pw_blk_cols), dtype=nb_dtype)
        temp_shr  = cuda.shared.array(shape=(pw_blk_rows, pw_blk_cols), dtype=nb_dtype)

        if (p < N) and (w < W):
            c0 = 0.0
            c1 = 0.0
            c2 = 0.0
            if shape <= -0.5:
                # I'm an ignore wall
                pw_dt_shr[row_loc, col_loc] = np.inf
            else:
                point  = cuda.shared.array(shape=dim, dtype=nb_dtype)
                normal = cuda.shared.array(shape=dim, dtype=nb_dtype)
                d = row_loc
                if d < dim:
                    point[d]  = walls[w, 1, d]
                    normal[d] = walls[w, 2, d]
                cuda.syncthreads()
                if shape <= 0.5:
                    # I'm a flat wall
                    for d in range(dim):
                        dx = pos_loc[p, d] - point[d]
                        dv = vel[p, d]
                        c0 += dx * normal[d]
                        c1 += dv * normal[d]
                    c0 -= pw_gap_min[p, w]
                elif shape <= 1.5:
                    # I'm a sphere wall
                    for d in range(dim):
                        dx = pos_loc[p, d] - point[d]
                        dv = vel[p, d]
                        c0 += dx * dx
                        c1 += dx * dv * 2
                        c2 += dv * dv
                    c0 -= pw_gap_min[p, w]**2
                elif shape <= 2.5:
                    # I'm a cylinder wall
                    dx_ax_mag = 0.0
                    dv_ax_mag = 0.0
                    for d in range(dim):
                        dx = pos_loc[p, d] - point[d]
                        dv = vel[p, d]
                        dx_ax_mag += dx * normal[d]
                        dv_ax_mag += dv * normal[d]

                    for d in range(dim):
                        dx = pos_loc[p, d] - point[d]
                        dv = vel[p, d]
                        dx_ax = dx_ax_mag * normal[d]
                        dx_normal = dx - dx_ax
                        dv_ax = dv_ax_mag * normal[d]
                        dv_normal = dv - dv_ax
                        c0 += dx_normal * dx_normal
                        c1 += dx_normal * dv_normal * 2
                        c2 += dv_normal * dv_normal
                    c0 -= pw_gap_min[p,w]**2
                else:
                    raise Exception('Invalid wall type')
                dt = solver_gpu(c2, c1, c0, pw_events_dev[p, w])
        else:
            dt = np.inf

        pw_dt_shr[row_loc, col_loc] = dt
        temp_shr[row_loc, col_loc] = dt
        cuda.syncthreads()
        min_gpu(temp_shr)
        min_dt = temp_shr[0, 0]
        
        if (p < N) and (w < W):
            pw_events_dev[p, w] = (dt < min_dt + thresh)
            if idx_loc == 0:
                pw_dt_blk_dev[blk_row, blk_col] = min_dt
            if check_gpu_cpu:
                pw_dt_dev[p, w] = dt



    def get_pp_dt_gpu(part, walls):
        get_pp_dt_kernel[pp_grd_shp, pp_blk_shp](part.pos_dev, part.vel_dev, part.pp_gap_min_dev, part.pp_events_dev, part.pp_dt_blk_dev, part.pp_dt_dev)
        
        if check_gpu_cpu:
            part.pp_dt_gpu = part.pp_dt_dev.copy_to_host()

        part.pp_events_new = part.pp_events_dev.copy_to_host()
        part.pp_dt_blk_gpu = part.pp_dt_blk_dev.copy_to_host()
        
        part.pp_dt = np.min(part.pp_dt_blk_gpu)  # min over blocks
        blk_idx = (part.pp_dt_blk_gpu < part.pp_dt + thresh)  # which blocks obtained that min
        
        # Need to remove True for p-p dts that were smaller than all others on their block
        # but are larger than the global dt from other blocks.   The corresponding
        # entry of blk_idx is False and part.pp_events_new is True.
        # So, we reshape blk_idx and take the entry-wise AND.

        reps = np.prod(pp_blk_shp)  # number of threads on each block
        A = np.repeat(blk_idx.ravel(), reps)  # repetition for each thread on the block
        new_sh = part.pp_events_new.shape
        blk_events = A[:np.prod(new_sh)].reshape(new_sh)  # drop extra entries and reshape
        part.pp_events_new &= blk_events


    def get_pw_dt_gpu(part, walls):
        ## See comments for get_pp_dt_gpu
        
        get_pw_dt_kernel[pw_grd_shp, pw_blk_shp](part.pos_loc_dev, part.vel_dev, part.pw_gap_min_dev, part.walls_data_dev, part.pw_events_dev, part.pw_dt_blk_dev, part.pw_dt_dev)
        
        if check_gpu_cpu:
            part.pw_dt_gpu = part.pw_dt_dev.copy_to_host()

        part.pw_events_new = part.pw_events_dev.copy_to_host()
        part.pw_dt_blk_gpu = part.pw_dt_blk_dev.copy_to_host()
        part.pw_dt = np.min(part.pw_dt_blk_gpu)
        blk_idx = (part.pw_dt_blk_gpu < part.pw_dt + thresh)
        
        reps = np.prod(pw_blk_shp)
        A = np.repeat(blk_idx.ravel(), reps)
        new_sh = part.pw_events_new.shape
        blk_events = A[:np.prod(new_sh)].reshape(new_sh)
        part.pw_events_new &= blk_events