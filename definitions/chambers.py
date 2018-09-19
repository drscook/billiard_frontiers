def parallel_planes(width):
    walls = []
    walls.append(FlatWall(base_point = [0, width, 0], normal = [0, -1, 0], tangents = [[1, 0, 0], [0, 0, 1]], side='inside'))
    walls.append(FlatWall(base_point = [0, -width, 0], normal = [0, 1, 0], tangents = [[1, 0, 0], [0, 0, 1]], side='inside'))
    
    return walls

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
    return walls

def sinai(cell_size, scatter_radius):
    cell_size = np.asarray(cell_size, dtype=np_dtype)
    if np.any(scatter_radius > cell_size):
        raise Exception('scatterer larger than box')
    walls = box(cell_size)
    base_point = 0.0 * walls[0].base_point
    walls.append(SphereWall(base_point=base_point, radius=scatter_radius, side='outside'))
    return walls
    
def lorentz_rectangle(cell_size, scatter_radius):
    walls = sinai(cell_size, scatter_radius)
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
    return walls

def lorentz_hexagonal(scatter_radius, part_radius, horizon_factor):
    # horizon_factor < 1 for finite horizon, horizon_factor > 1 for infinite horizon
    R = scatter_radius + part_radius
    gap_crit = (2/np.sqrt(3) - 1) * R
    gap = horizon_factor * gap_crit
    x0 = R + gap
    y0 = np.sqrt(3) * x0
    cell_size = np.array([x0,y0])
    print('Diameter / spacing = {:.3f} / {:.3f} = {:.3f}'.format(2*R, 2*x0, (2*R)/(2*x0)))
    
    walls = lorentz_rectangle(cell_size, scatter_radius)
    walls.append(SphereWall(base_point=np.array([x0,y0]), radius=scatter_radius, side='outside'))
    walls.append(SphereWall(base_point=np.array([-x0,y0]), radius=scatter_radius, side='outside'))
    walls.append(SphereWall(base_point=np.array([-x0,-y0]), radius=scatter_radius, side='outside'))
    walls.append(SphereWall(base_point=np.array([x0,-y0]), radius=scatter_radius, side='outside'))
    return walls, cell_size


def cylinder(base_point, radius, cylinder_axis, radial_direction, angle):
    if dim != 3:
        raise Exception('Cylinder chamber must be dimension 3')
    walls = []
    walls.append(CylinderWall(base_point = base_point, radius = radius, 
                             cylinder_axis = cylinder_axis, 
                             radial_direction = radial_direction, angle = angle
                             , side='inside'))
    return walls