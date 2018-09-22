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
    cuda.to_device(part.pp_mask, to=part.pp_mask_dev)
    cuda.to_device(part.pw_mask, to=part.pw_mask_dev)
    

def init_gpu(part, walls):
    global dim
    global pp_blk_rows, pp_blk_cols, pp_blk_shp
    global pp_grd_rows, pp_grd_cols, pp_grd_shp
    global pw_blk_rows, pw_blk_cols, pw_blk_shp
    global pw_grd_rows, pw_grd_cols, pw_grd_shp
    global get_pp_dt_gpu, get_pw_dt_gpu
    
    dim = part.dim

    bp = min(optimal_part_num, part.num)
    pp_blk_rows = bp
    pp_blk_cols = bp

    gp = int(np.ceil(part.num / bp))
    pp_grd_rows = gp
    pp_grd_cols = gp

    bw = min(threads_per_block_max, part.num)
    pw_blk_rows = bw
    pw_blk_cols = 1

    gw = int(np.ceil(part.num / bw))
    pw_grd_rows = gw
    pw_grd_cols = len(walls)
    
    pp_blk_shp = (pp_blk_rows, pp_blk_cols)
    pp_grd_shp = (pp_grd_rows, pp_grd_cols)
    pw_blk_shp = (pw_blk_rows, pw_blk_cols)
    pw_grd_shp = (pw_grd_rows, pw_grd_cols)


    part.walls_data = np.array([wall.data.copy() for wall in walls], dtype=np_dtype)
    part.walls_data_dev = cuda.to_device(part.walls_data)

    part.pp_gap_min_dev = cuda.to_device(part.pp_gap_min)
    part.pw_gap_min_dev = cuda.to_device(part.pw_gap_min)

    part.pos_dev = cuda.to_device(part.pos)
    part.pos_loc_dev = cuda.to_device(part.pos_loc)
    part.vel_dev = cuda.to_device(part.vel)
    part.pp_mask_dev = cuda.to_device(part.pp_mask)
    part.pw_mask_dev = cuda.to_device(part.pw_mask)

    part.pp_dt_gpu_block = cuda.pinned_array([part.num, pp_grd_cols], dtype=np_dtype)
    part.pp_dt_dev = cuda.to_device(part.pp_dt_gpu_block)

    part.pw_dt_gpu       = cuda.pinned_array([part.num, pw_grd_cols], dtype=np_dtype)
    part.pw_dt_dev = cuda.to_device(part.pw_dt_gpu)

    update_gpu(part)


    @cuda.jit(device=False)
    def get_pp_dt_kernel(pos, vel, pp_mask, pp_gap_min, pp_dt_dev):#, pp_col_coefs):
        pp_dt_shr = cuda.shared.array(shape=(pp_blk_rows, pp_blk_cols), dtype=nb_dtype)
        pos1_shr  = cuda.shared.array(shape=(pp_blk_rows, dim), dtype=nb_dtype)
        pos2_shr  = cuda.shared.array(shape=(pp_blk_rows, dim), dtype=nb_dtype)
        vel1_shr  = cuda.shared.array(shape=(pp_blk_rows, dim), dtype=nb_dtype)
        vel2_shr  = cuda.shared.array(shape=(pp_blk_rows, dim), dtype=nb_dtype)

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        p = tx + bx * cuda.blockDim.x
        q = ty + by * cuda.blockDim.y
        N = pos.shape[0]
        if ty < dim:
            pos1_shr[tx,ty] = pos[p,ty]
            vel1_shr[tx,ty] = vel[p,ty]
        if tx < dim:
            pos2_shr[ty,tx] = pos[q,tx]
            vel2_shr[ty,tx] = vel[q,tx]
        cuda.syncthreads()

        if (p < N) and (q < N):
            c0 = 0.0
            c1 = 0.0
            c2 = 0.0
            for d in range(dim):
                dx = pos1_shr[tx,d] - pos2_shr[ty,d]
                dv = vel1_shr[tx,d] - vel2_shr[ty,d]
                c0 += (dx**2)
                c1 += (dx*dv*2)
                c2 += (dv**2)
            c0 -= (pp_gap_min[p,q]**2)
            pp_dt_shr[tx,ty] = solver_gpu(c2, c1, c0, pp_mask[p,q])
        else:
            pp_dt_shr[tx,ty] = np.inf

        row_min_gpu(pp_dt_shr)
        if p < N:
            pp_dt_dev[p, by] = pp_dt_shr[tx,0]


    @cuda.jit(device=False)
    def get_pw_dt_kernel(pos_loc, vel, pw_mask, pw_gap_min, walls, pw_dt_dev):
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        p = tx + bx * cuda.blockDim.x
        w = ty + by * cuda.blockDim.y
        N = pos_loc.shape[0]
        W = walls.shape[0]
        shape = walls[w,0,0]

        if (p < N) and (w < W):
            c0 = 0.0
            c1 = 0.0
            c2 = 0.0
            if shape <= -0.5:
                # I'm an ignore wall
                pw_dt_dev[p,w] = np.inf
            else:
                point = cuda.shared.array(shape=dim, dtype=nb_dtype)
                vec = cuda.shared.array(shape=dim, dtype=nb_dtype)
                if tx < dim:
                    point[tx] = walls[w,1,tx]
                    vec[tx] = walls[w,2,tx]
                cuda.syncthreads()
                if shape <= 0.5:
                    # I'm a flat wall
                    for d in range(dim):
                        dx = pos_loc[p,d] - point[d]
                        dv = vel[p,d]
                        c0 += dx * vec[d]
                        c1 += dv * vec[d]
                    c0 -= pw_gap_min[p,w]
                elif shape <= 1.5:
                    # I'm a sphere wall
                    for d in range(dim):
                        dx = pos_loc[p,d] - point[d]
                        dv = vel[p,d]
                        c0 += dx * dx
                        c1 += dx * dv * 2
                        c2 += dv * dv
                    c0 -= pw_gap_min[p,w]**2
                elif shape <= 2.5:
                    # I'm a cylinder wall
                    dx_ax_mag = 0.0
                    dv_ax_mag = 0.0
                    for d in range(dim):
                        dx = pos_loc[p,d] - point[d]
                        dv = vel[p,d]
                        dx_ax_mag += dx * vec[d]
                        dv_ax_mag += dv * vec[d]

                    for d in range(dim):
                        dx = pos_loc[p,d] - point[d]
                        dv = vel[p,d]
                        dx_ax = dx_ax_mag * vec[d]
                        dx_normal = dx - dx_ax
                        dv_ax = dv_ax_mag * vec[d]
                        dv_normal = dv - dv_ax
                        c0 += dx_normal * dx_normal
                        c1 += dx_normal * dv_normal * 2
                        c2 += dv_normal * dv_normal
                    c0 -= pw_gap_min[p,w]**2

                else:
                    raise Exception('Invalid wall type')
                pw_dt_dev[p,w] = solver_gpu(c2, c1, c0, pw_mask[p,w])  

    
    def get_pp_dt_gpu(part, walls):
        get_pp_dt_kernel[pp_grd_shp, pp_blk_shp](part.pos_dev, part.vel_dev,
            part.pp_mask_dev, part.pp_gap_min_dev, part.pp_dt_dev)
        part.pp_dt_gpu_block = part.pp_dt_dev.copy_to_host()
        part.pp_dt_gpu = np.min(part.pp_dt_gpu_block, axis=-1)
        return part.pp_dt_gpu


    def get_pw_dt_gpu(part, walls):
        get_pw_dt_kernel[pw_grd_shp, pw_blk_shp](part.pos_loc_dev,
            part.vel_dev, part.pw_mask_dev, part.pw_gap_min_dev, part.walls_data_dev,
            part.pw_dt_dev)
        part.pw_dt_gpu = part.pw_dt_dev.copy_to_host()
        return part.pw_dt_gpu