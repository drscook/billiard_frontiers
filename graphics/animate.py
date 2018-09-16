from matplotlib import animation, rc
import io
import base64
from IPython.display import HTML

def read_and_interpolate(date=None, run=None, max_steps=None):
    import scipy.linalg
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
    with tables.open_file(data_filename, mode='r') as data_file:
        x = data_file.root['pos'][:max_steps]
        v = data_file.root['vel'][:max_steps]
        s = data_file.root['spin'][:max_steps]
        t = data_file.root['t'][:max_steps]

    dts = np.diff(t)
    dt_median = np.percentile(dts, 50)
    short_step = dts < (dt_median / 1000)
    nominal_frame_length = np.percentile(dts[~short_step], 10)
    num_frames = np.round(dts / nominal_frame_length).astype(int) # Divide each step into pieces of length as close to nominal_frame_length as possible
    num_frames[num_frames<1] = 1
    ddts = dts / num_frames  # Compute frame length within each step

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
        do = [scipy.linalg.expm(ddt * U) for U in re_s[-1]] # incremental rotatation during each frame
        for f in range(num_frames[i]):
            re_t.append(re_t[-1] + ddt)
            re_x.append(re_x[-1] + dx)
            re_v.append(re_v[-1])
            re_s.append(re_s[-1])
#             B = [A.dot(Z) for (A,Z) in zip(re_o[-1], do)] # rotates each particle the right amount
            B = np.einsum('pde,pef->pdf', re_o[-1], do)  # more efficient version of calculation above
            re_o.append(B)
    
    data = {'t': np.asarray(re_t), 'raw_t': np.asarray(t)
           ,'pos': np.asarray(re_x) ,'raw_pos': np.asarray(x)
           ,'vel': np.asarray(re_x) ,'raw_vel': np.asarray(v)
           ,'spin': np.asarray(re_s) ,'raw_spin': np.asarray(s)
           ,'orient': np.asarray(re_o)}

    return part_params, wall_params, data


translates = np.array([[0.0,0.0]])

def animate(part_params, wall_params, data, run_time=20):
    t = data['t']
    x = data['pos']
    o = data['orient']
    mesh = np.asarray(part_params['mesh'])
    clr = part_params['clr']
#     print(clr.shape)
    
    frame_num, part_num, dim = x.shape
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    for trans in translates:
        for w in wall_params:
            ax.plot(*((w['mesh']+trans).T), color='black')

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    path = []
    bdy = []
    for p in range(part_num):
        path.append(ax.plot([],[], color=clr[p])[0])
        bdy.append(ax.plot([],[], color=clr[p])[0])

    def init():
        time_text.set_text('')
        for p in range(part_num):
            path[p].set_data([], [])
            bdy[p].set_data([], [])
        return path + bdy

    def update(s):
        time_text.set_text(f"step {s}, time {t[s]:.2f}")
#         step_text.set_text('time = %.1f' % pendulum.time_elapsed)
        for p in range(part_num):
            path[p].set_data(*(x[:s+1,p].T))
            bdy[p].set_data(*((mesh[p].dot(o[s,p].T) + x[s,p]).T))
        return path + bdy
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=frame_num, interval=run_time*1000/frame_num, blit=True)
    plt.close()
    return anim

def play_video(fname):
    video = io.open(fname, 'r+b').read()
    encoded = base64.b64encode(video)

    display(HTML(data='''<video alt="test" controls>
         <source src="data:video/mp4;base64,{0}" type="video/mp4" />
         </video>'''.format(encoded.decode('ascii'))))

