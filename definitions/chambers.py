##################################################################
### Recently Tested ###
##################################################################


def box(cell_size):
    cell_size = np.asarray(cell_size, dtype=np_dtype)
    Tangents = np.diag(cell_size)
    walls = []
    for d, L in enumerate(cell_size):
        for s in [-1,1]:
            base_point = 0.0 * cell_size
            base_point[d] = s*L
            walls.append(FlatWall(base_point=base_point.copy(), normal=-base_point.copy()
                                   ,tangents = np.delete(Tangents,d,0)))
    return walls, cell_size

def sinai(cell_size, scatter_radius):
    cell_size = np.asarray(cell_size, dtype=np_dtype)
    if np.any(scatter_radius > cell_size):
        raise Exception(f"scatter radius {scatter_radius} larger than cell size {cell_size}")
    walls = box(cell_size)
    base_point = 0.0 * walls[0].base_point
    walls.append(SphereWall(base_point=base_point, radius=scatter_radius, side='outside'))
    return walls, cell_size


def sinai_double(cell_size, scatter_radius):
    cell_size = np.asarray(cell_size, dtype=np_dtype)
    cell_size[0] *= 2
    
    if np.any(scatter_radius > cell_size):
        raise Exception(f"scatter radius {scatter_radius} larger than cell size {cell_size}")
    walls, cell_size = box(cell_size)
    base_point = np.array([cell_size[1], 0])
    walls.append(SphereWall(base_point=   base_point, radius=scatter_radius, side='outside'))
    walls.append(SphereWall(base_point=-1*base_point, radius=scatter_radius, side='outside'))
    return walls, cell_size


def lorentz_rectangle(cell_size, scatter_radius):
    walls, cell_size = sinai(cell_size, scatter_radius)
    s = -1
    for (w, wall) in enumerate(walls[:4]):
        wall.pw_gap_m = 0.0
        wall.pw_gap_b = 0.0
        d = int(np.floor(w/2))
        s *= -1
        wall.pw_collision_law = PW_PeriodicLaw()
        wall.wrap_dim = d
        wall.wrap_wall_idx = w+s
        wall.clr = 'clear'
    return walls, cell_size


def lorentz_hexagonal(scatter_radius, part_radius, horizon_factor):
    """
    trapped/overlap: hf < 0; finite horizon: 0 <= hf < 1; infinite horizon: 1 <= hf < inf
    """    
    if horizon_factor < 0:
        raise Exception(f"horizon factor must be >= 0, but given {horizon_factor}")
    R = scatter_radius + part_radius
    b = np.sqrt(3) / 2
    x_crit = R / b
    
    # The code below could be confusing because it involves 3 variables essentially redundant variables.
    # The most important is k.  The key definition is x = x_crit * k.
    # However, the ranges for k for trapped/finite horizon/infinite horizon are a bit tricky to recall.
    # So we provide a convenient rescaled version called horizon_factor.
    # We also compute a physically natural variable called linear density, mostly for output.
    # These are all essentially redundant, but serve different purposes.
    # They are defined below with associated scales.  Recall that b = sqrt(3)/2
    # horizon factor = (k - b) / (1 - b)
    # k = horizon factor * (1 - b) + b
    # linear density ld = x / R
    # trapped/overlap: hf < 0; finite horizon: 0 <= hf < 1; infinite horizon: 1 <= hf < inf
    # trapped/overlap: k  < b; finite horizon: b <= k  < 1; infinite horizon: 1 <= b  < inf
    # trapped/overlap: ld > 1; finite horizon: b <= ld < 1; infinite horizon: 0 <= ld < b
    
    k = horizon_factor * (1 - b) + b
    x = x_crit * k
    y = 2 * x * b
    cell_size = np.array([x,y])
    
    print(f"\nlinear density = effective radius / spacing = {R:.3f} / {x:.3f} = {(R / x):.3f}")
    print(f"For context, here are horzon ranges in terms of linear density.")
    print(f"infinite horizon: [0,{b:.3f}]; finite horizon: ({b:.3f},1]; trapped/overlap: (1, inf)\n")
    print("Don't forget to pass part_radius to Particles.  Otherwise it will use the default.")
    
    walls, cell_size = lorentz_rectangle(cell_size, scatter_radius)
    walls.append(SphereWall(base_point=np.array([x,y]), radius=scatter_radius, side='outside'))
    walls.append(SphereWall(base_point=np.array([-x,y]), radius=scatter_radius, side='outside'))
    walls.append(SphereWall(base_point=np.array([-x,-y]), radius=scatter_radius, side='outside'))
    walls.append(SphereWall(base_point=np.array([x,-y]), radius=scatter_radius, side='outside'))

    return walls, cell_size



def parallel_lines(cell_size):
    cell_size = np.asarray(cell_size, dtype=np_dtype)
    walls = []
    walls.append(FlatWall(base_point = [0, cell_size[1]], normal = [0, -1], tangents = [[cell_size[0], 0]]))
    walls.append(FlatWall(base_point = [0, -cell_size[1]], normal = [0, 1], tangents = [[cell_size[0], 0]]))
    return walls, cell_size



##################################################################
### Not Recently Tested - may be out of date ###
##################################################################


def parallel_s(width):
    cell_size = np.asarray([width, width], dtype=np_dtype)
    walls = []
    walls.append(FlatWall(base_point = [0, width, 0], normal = [0, -1, 0], tangents = [[1, 0, 0], [0, 0, 1]]))
    walls.append(FlatWall(base_point = [0, -width, 0], normal = [0, 1, 0], tangents = [[1, 0, 0], [0, 0, 1]]))
    return walls, cell_size



def non_parallel_planes(width):
    walls = []
    eps=0.001
    walls.append(FlatWall(base_point = [0, width, 0], normal = [0, -1, 0], tangents = [[1, 0, 0], [0, 0, 1]]))
    walls.append(FlatWall(base_point = [0, -width, 0], normal = [math.sin(eps), math.cos(eps), 0], tangents = [[math.cos(eps), -math.sin(eps), 0], [0, -math.sin(eps), math.cos(eps)]]))
    
    return walls

# def sphereplane(width):
#     walls = []
#     walls.append(FlatWall(base_point = [0, width, 0], normal = [0, -1, 0], tangents = [[1, 0, 0], [0, 0, 1]]))
#     walls.append(SphereWall(base_point = [0,-100*width,0], radius=99*width))
    
#     return walls

def rectangular_channel(length=10, width=10):
    walls = [FlatWall(base_point=[length,0,0], normal=[-1,0,0], tangents=[[0,1,0],[0,0,1]])
           ,FlatWall(base_point=[-length,0,0], normal=[1,0,0], tangents=[[0,1,0],[0,0,1]])
           ,FlatWall(base_point=[0,width,0], normal=[0,-1,0], tangents=[[1,0,0],[0,0,1]])
           ,FlatWall(base_point=[0,-width,0], normal=[0,1,0], tangents=[[1,0,0],[0,0,1]])
           ]
    return walls





def cylinder(base_point, radius, cylinder_axis, radial_direction, angle):
    if dim != 3:
        raise Exception('Cylinder chamber must be dimension 3')
    walls = []
    walls.append(CylinderWall(base_point = base_point, radius = radius, 
                             cylinder_axis = cylinder_axis, 
                             radial_direction = radial_direction, angle = angle
                             , side='inside'))
    return walls