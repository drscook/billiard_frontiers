# master wall class; subclass for each wall shape
class Wall():
    def __init__(self, dim, base_point, side='outside'):
        self.dim = dim
        self.base_point = np.asarray(base_point, dtype=np_dtype)
        self.temp = 1.0
        self.pw_collision_law = PW_SpecularLaw()
        self.side = side
        if side == 'outside':
            self.sign = 1
        elif side == 'inside':
            self.sign = -1
        else:
            raise Exception('Invalid side - must be inside or outside')
        self.pw_gap_m = self.sign
        self.pw_gap_b = 0.0
        self.data = np.full([3, dim], np.inf, dtype=np_dtype)
        self.data[1,:] = self.base_point.copy()
        self.record_params = ['dim', 'base_point', 'temp', 'side', 'name', 'mesh']
    
    def params(self):
        def f(x):
            try:
                return x.tolist()
            except:
                return x
        d = {param:f(getattr(self, param)) for param in self.record_params}
        d['pw_collision_law'] = self.pw_collision_law.name            
        return d
    
    @staticmethod
    def get_mesh():
        raise Exception('You should implement the method get_mesh() in a subclass.')

    def get_pw_gap(self):
        return self.get_pw_col_coefs(gap_only=True)
    
    def resolve_pw_collision(self, part, walls, p):
        self.pw_collision_law.resolve_collision(part=part, walls=walls, p=p, w=self.idx)

    @staticmethod
    def normal(pos):
        raise Exception('You should implement the method normal() in a subclass.')

    @staticmethod
    def get_pw_col_coefs(self):
        raise Exception('You should implement the method get_pw_col_coefs() in a subclass.')
        
        
        
        
        
class FlatWall(Wall):
    def __init__(self, dim, base_point, normal, tangents):
        super().__init__(dim, base_point)
        self.name = 'flat'
        self.normal_static = make_unit(normal)
        self.tangents = np.asarray(tangents, dtype=np_dtype)
        self.data[0,0] = 0
        self.data[2,:] = self.normal_static.copy()
        self.record_params.extend(['normal_static', 'tangents'])
        
    def normal(self, pos):
        return self.normal_static
    
    def get_pw_col_coefs(self, gap_only=False):
        dx = part.pos_loc - self.base_point
        nu = self.normal_static
        c0 = dx.dot(nu) - self.pw_gap_min
        c0[np.isinf(c0)] = np.inf #corrects -np.inf to +np.inf
        c0[np.isnan(c0)] = np.inf #corrects np.nan to +np.inf
        if gap_only == True:
            return c0
        dv = part.vel
        c1 = dv.dot(nu)
#         c2 = np.zeros(c0.shape, dtype=np_dtype)
#         self.pw_col_coefs = np.array([c2, c1, c0]).T
        self.pw_col_coefs = np.array([c1, c0]).T
        return self.pw_col_coefs
    
    def get_mesh(self):
        self.mesh = flat_mesh(self.tangents) + self.base_point  # see visualize.py


        
        
        
        
        
        
class SphereWall(Wall):
    def __init__(self, dim, base_point, radius, side='outside'):
        super().__init__(dim, base_point, side)
        self.name = 'sphere'
        self.radius = radius
        self.pw_gap_b = radius
        self.data[0,0] = 1
        self.record_params.extend(['radius'])

    def normal(self, pos):
        dx = pos - self.base_point
        return make_unit(dx) * self.sign  # sign flips if on inside

    def get_pw_col_coefs(self, gap_only=False):
        dx = part.pos_loc - self.base_point
        c0 = contract(dx*dx)
        if gap_only == True:
            return (np.sqrt(c0) - self.pw_gap_min)*self.sign
        c0 -= self.pw_gap_min**2
        dv = part.vel
        c1 = contract(dx*dv) * 2
        c2 = contract(dv**2)
        self.pw_col_coefs = np.array([c2, c1, c0]).T
        return self.pw_col_coefs

    def get_mesh(self):
        self.mesh = sphere_mesh(self.dim, self.radius) + self.base_point # see visualize.py

        
        
        
        
        
class CylinderWall(Wall):
    def __init__(self, dim, base_point, radius, cylinder_axis, radial_direction, angle=2*np.pi, side='inside'):
        super().__init__(dim, base_point, side)
        self.name = 'cylinder'
        self.pw_gap_b = radius
        self.radius = radius
        self.cylinder_axis = np.asarray(cylinder_axis, dtype=np_dtype)
        self.axial_direction = make_unit(cylinder_axis)                                                                             
        self.radial_direction = make_unit(radial_direction)
        self.angle = angle
        self.data[0,0] = 2
        self.data[2,:] = self.axial_direction.copy()
    
    def normal(self, pos):
        dx = pos - self.base_point
        ax = self.axial_direction
        w = dx.dot(ax) * ax
        n = dx - w
        return make_unit(n)*self.sign
  
    def get_pw_col_coefs(self, gap_only=False):
        dx = part.pos_loc - self.base_point
        ax = self.axial_direction
        dx_ax = dx.dot(ax)[:,np.newaxis] * ax
        dx_normal = dx - dx_ax
        c0 = contract(dx_normal**2)
        if gap_only == True:
            return (np.sqrt(c0) - self.pw_gap_min) * self.sign
        c0 -= self.pw_gap_min**2
        dv = part.vel
        dv_ax = dv.dot(ax)[:,np.newaxis] * ax
        dv_normal = dv - dv_ax
        c1 = contract(dx_normal*dv_normal) * 2
        c2 = contract(dv_normal**2)
        self.pw_col_coefs = np.array([c2, c1, c0]).T
        return self.pw_col_coefs