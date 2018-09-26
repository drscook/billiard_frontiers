from matplotlib import animation, rc
import io
import base64
from IPython.display import HTML

def find_records(date=None, run=None):
    if date is None:
        date = str(datetime.date.today())
    date_path = root_path + date + '/'
    
    if run is None:
        run = get_last_file_in_dir(date_path)
    run_path = date_path + str(run) + '/'
    
    part_params_filename = run_path + 'part_params.json'
    with open(part_params_filename, mode='r') as part_params_file:
        part_params = json.load(part_params_file)

    wall_params_filename = run_path + 'wall_params.json'
    with open(wall_params_filename, mode='r') as wall_params_file:
        wall_params = json.load(wall_params_file)

    data_filename = run_path + 'data.hdf5'
    return part_params, wall_params, data_filename, run_path


def read_and_interpolate(data_filename, frame_min=None, distortion_max=None, compute_orient=False):
    import scipy.linalg
    
    with tables.open_file(data_filename, mode='r') as data_file:
        x = np.asarray(data_file.root['pos'])
        v = np.asarray(data_file.root['vel'])
        s = np.asarray(data_file.root['spin'])
        t = np.asarray(data_file.root['t'])

    if distortion_max is not None:
        def remove_short(x):
            median = np.percentile(x, 50)
            short_step = x < (median / 10000)
            return x[~short_step]

        dts = np.diff(t)
        cutoff = False
        for rank in np.linspace(100,0,50):
            nominal_frame_length = np.percentile(remove_short(dts), rank)

            frames_per_step = dts / nominal_frame_length # Divide each step into pieces of length as close to nominal_frame_length as possible
            k = frame_min / np.sum(frames_per_step)  # Divide each step into more pieces to achieve frames_min; ensures desired frame_rate_min
            frames_per_step *= k
            frames_per_step = np.round(frames_per_step).astype(int)
            frames_per_step[frames_per_step<1] = 1
            ddts = dts / frames_per_step  # Compute frame length within each step

            std = remove_short(ddts).std()
            mean = remove_short(ddts).mean()
            distortion = std / mean
            mes = f"rank cutoff = {rank:.0f} -> distortion = {distortion:.2f}"
            if distortion < distortion_max:
                print(f"{mes} < {distortion_max:.2f} -> that will work!!")
                break
    #         else:
    #             print(f"{mes} >= {distortion_max:.2f} -> use a tighter rank cutoff")


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
           ,'orient': np.asarray(re_o)}
    data['step_num'], data['part_num'], data['dim'] = data['pos'].shape
    
    return part_params, wall_params, data


def get_cell_translates(pos, cell_size):
    dim = len(cell_size)
    cs = np.asarray(cell_size) * 2
    m = (pos.min(axis=0).min(axis=0) / cs).round()
    M = (pos.max(axis=0).max(axis=0) / cs).round()
    z = [np.arange(m[d], M[d]+1) * cs[d] for d in range(dim)]
    translates = it.product(*z)
    return [np.asarray(t) for t in translates]



def animate(part_params, wall_params, data, movie_time=20, frame_max=None, show_trails=True):
    assert hasattr(data, 'orient'), "Must run read_and_interpolate with compute_orient=True"
        

    frame_num = len(data['t'])
    print(frame_num)
    if frame_max is None:
        frame_max = frame_num

    if frame_num > frame_max:
        print(f"Cutting movie short to satify frame_max.  Consider increasing anim_time or distortion_max.")
        frame_num = frame_max
        
    t = data['t'][:frame_num]
    x = data['pos'][:frame_num]
    o = data['orient'][:frame_num]
    mesh = np.asarray(part_params['mesh'])
    clr = part_params['clr']

    cell_translates = get_cell_translates(x, part_params['cell_size'])
    
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
                                   frames=frame_num, interval=movie_time*1000/frame_num, blit=True)
    plt.close()
    return anim

def play_video(fname):
    video = io.open(fname, 'r+b').read()
    encoded = base64.b64encode(video)

    display(HTML(data='''<video alt="test" controls>
         <source src="data:video/mp4;base64,{0}" type="video/mp4" />
         </video>'''.format(encoded.decode('ascii'))))

