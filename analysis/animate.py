from matplotlib import animation, rc
import io
import base64
from IPython.display import HTML
import scipy.linalg

def interpolate(data_filename, frame_min=1, frame_max=None, distortion_max=None, compute_orient=True):
    data_file = tables.open_file(data_filename, mode='r')
    t = data_file.root['t']
    dts = np.diff(t)
    
    def get_cutoff(frames_per_step, frame_max=None):
        cutoff = None
        if frame_max is not None:
            frame_num = np.sum(frames_per_step)
            if frame_num > frame_max:
                frame_cum = np.cumsum(frames_per_step)
                cutoff = np.argmax(frame_cum > frame_max)
        return cutoff

    
    if distortion_max is None:
        ddts = dts
        frames_per_step = np.ones_like(dts).astype(int)
    else:
        def remove_short(x):
            median = np.percentile(x, 50)
            short_step = x < (median / 1000)
            return x[~short_step]

        median_step_length = np.percentile(remove_short(dts), 50)        
        for rank in np.linspace(1.0, 0.0, 50):
            nominal_frame_length = median_step_length * rank
            frames_per_step = dts / nominal_frame_length # Divide each step into pieces of length as close to nominal_frame_length as possible
            k = max(frame_min / np.sum(frames_per_step), 1.0)  # Divide each step into more pieces to achieve frames_min; ensures desired frame_rate_min            
            frames_per_step *= k
            frames_per_step = np.round(frames_per_step).astype(int)
            frames_per_step[frames_per_step<1] = 1
            ddts = dts / frames_per_step  # Compute frame length within each step
            
            cutoff = get_cutoff(frames_per_step, frame_max=frame_max)
            if cutoff is not None:
#                 dts = dts[:cutoff]
                frames_per_step = frames_per_step[:cutoff]
                ddts = ddts[:cutoff]
#                 print(f"Cutting movie short after {cutoff} collisions to satify frame_max.  Consider increasing frame_max via anim_time or distortion_max.")
            rs = remove_short(ddts)
            distortion = rs.std() / rs.mean()
            mes = f"rank cutoff = {rank:.2f} -> distortion = {distortion:.2f}"
            if distortion < distortion_max:
                print(f"{mes} < {distortion_max:.2f} -> that will work!!")
                break
#             else:
#                 print(f"{mes} >= {distortion_max:.2f} -> use a tighter rank cutoff")

        

    cutoff = get_cutoff(frames_per_step, frame_max=frame_max)
    if cutoff is not None:
        frames_per_step = frames_per_step[:cutoff]
        ddts = ddts[:cutoff]
        print(f"Cutting movie short after {cutoff} collisions to satify frame_max.  Consider increasing frame_max via anim_time or distortion_max.")

        
#     with tables.open_file(data_filename, mode='r') as data_file:
    t = np.asarray(data_file.root['t'][:cutoff]).astype(np.float32)
    x = np.asarray(data_file.root['pos'][:cutoff]).astype(np.float32)
    s = np.asarray(data_file.root['spin'][:cutoff]).astype(np.float32)
    
    with np.errstate(invalid='ignore', divide='ignore'): 
        v = np.diff(x, axis=0) / np.diff(t).reshape(-1,1,1)
    v = np.append(v, v[[-1]], axis=0)
    v[np.isinf(v)] = 0.0
    v[np.isnan(v)] = 0.0
        
    re_t, re_x, re_v, re_s = [t[0]], [x[0]], [v[0]], [s[0]]
    _, part_num, dim, _ = s.shape
    I = np.eye(dim, dtype=np_dtype)
    re_o = [np.repeat(I[np.newaxis], part_num, axis=0)]

    for (i, ddt) in enumerate(ddts):
        re_t[-1] = t[i]
        re_x[-1] = x[i]
        re_v[-1] = v[i]
        re_s[-1] = s[i]
        dx = re_v[-1] * ddt

        if compute_orient:
                do = [scipy.linalg.expm(ddt * U) for U in re_s[-1]] # incremental rotatation during each frame
        for f in range(frames_per_step[i]):
            re_t.append(re_t[-1] + ddt)
            re_x.append(re_x[-1] + dx)
            re_v.append(re_v[-1])
            re_s.append(re_s[-1])
            if compute_orient:
    #             B = [A.dot(Z) for (A,Z) in zip(re_o[-1], do)] # rotates each particle the right amount
                B = np.einsum('pde,pef->pdf', re_o[-1], do)  # more efficient version of calculation above
                re_o.append(B)
    
    data = {'t': np.asarray(re_t), 't_raw': np.asarray(t)
           ,'pos': np.asarray(re_x) ,'pos_raw': np.asarray(x)
           ,'vel': np.asarray(re_x) ,'vel_raw': np.asarray(v)
           ,'spin': np.asarray(re_s) ,'spin_raw': np.asarray(s)
           }
    data['frame_num'], data['part_num'], data['dim'] = data['pos'].shape
    try:
        data['orient'] = np.asarray(re_o)
        print("Orientation computed")
    except:
        print("Orientation not computed")
    data_file.close()
    return data


def get_cell_translates(pos, cell_size):
    dim = len(cell_size)
    cs = np.asarray(cell_size) * 2
    m = (pos.min(axis=0).min(axis=0) / cs).round()
    M = (pos.max(axis=0).max(axis=0) / cs).round()
    z = [np.arange(m[d], M[d]+1) * cs[d] for d in range(dim)]
    translates = it.product(*z)
    return [np.asarray(t) for t in translates]


def play_video(fname):
    video = io.open(fname, 'r+b').read()
    encoded = base64.b64encode(video)

    display(HTML(data='''<video alt="test" controls>
         <source src="data:video/mp4;base64,{0}" type="video/mp4" />
         </video>'''.format(encoded.decode('ascii'))))


def animate(date=None, run=None, show_trails=True, distortion_max=0.1, movie_time=20, frame_rate_min=20, frame_max=None, save=True, embed=False, dpi=None):
    start = timer()
    # To generate the movie, we must interpolate between collsion events.
    # Because the time between collision event varies, we get time distortion.
    # We can correct this distortion by adding more interpolated frames, but this increases animation time.
    # Here are several parameters to help the user balance distortion against animation time.
    # Setting both distortion_max and frame_max low will cut the movie short to accomodate.
    
    frame_min = movie_time * frame_rate_min
    part_params, wall_params, data_filename, run_path = find_records(date, run)

    data = interpolate(data_filename, frame_min=frame_min, frame_max=frame_max, distortion_max=distortion_max, compute_orient=True)
    print(f"I will attempt to animate {data['frame_num']} frames")
    
    t = data['t']
    x = data['pos']
    o = data['orient']
    mesh = np.asarray(part_params['mesh'])
    clr = part_params['clr']
    
#     cell_translates = get_cell_translates(x, part_params['cell_size'])
    cell_translates = [np.array([0,0])]
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid(False)
    fc = ax.get_facecolor()
    for w in wall_params:
        c = w['clr']
        if c == 'clear':
            pass
        else:
            for trans in cell_translates:
                ax.plot(*((w['mesh']+trans).T), color=c,  linewidth=1.0)

    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes)
    bdy = []
    trail = []
    for p in range(data['part_num']):
        bdy.append(ax.plot([],[], color=clr[p], linewidth=1.0)[0])        
        if show_trails:
            trail.append(ax.plot([],[], color=clr[p])[0])
        

    def init():
        time_text.set_text('')
        for p in range(data['part_num']):
            bdy[p].set_data([], [])
            if show_trails:
                trail[p].set_data([], [])
        return bdy + trail

    def update(s):
        time_text.set_text(f"time {t[s]:.2f}")
        for p in range(data['part_num']):
            bdy[p].set_data(*((mesh[p].dot(o[s,p].T) + x[s,p]).T))
            if show_trails:
                trail[p].set_data(*(x[:s+1,p].T))
        return bdy + trail
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=data['frame_num'], interval=movie_time*1000/data['frame_num'], 
                                   blit=True)
    plt.close()
    
    if save:
        anim_filename = run_path+'animation.mp4'
        anim.save(filename=anim_filename, dpi=dpi)    # save animation as mp4
        if embed:
            play_video(anim_filename)    # show in notebook - resizing issues
    elif embed:
        display(HTML(anim.to_jshtml()))        # diplays video in notebook

    elapsed = timer() - start
    anim_rate = data['frame_num'] / elapsed
    print(f"I animated {data['frame_num']} frames / {elapsed:.2f} sec = {anim_rate:.2f} frames / sec")
    
    return anim


