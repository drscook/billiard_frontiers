class PW_CollisionLaw:
    @staticmethod
    def resolve_collision(self, part, walls, p, w):
        raise Exception('You should implement the method resolve_collision() in a subclass.')

        
class PW_SpecularLaw(PW_CollisionLaw):
    name = 'PW_SpecularLaw'
    def resolve_collision(self, part, walls, p, w):
        nu = walls[w].normal(part.pos_loc[p])
        part.vel[p] -= 2 * part.vel[p].dot(nu) * nu

        
class PW_PeriodicLaw(PW_CollisionLaw):
    name = 'PW_PeriodicLaw'
    def resolve_collision(self, part, walls, p, w):
        w1 = w
        wall1 = walls[w1]
        w2 = wall1.wrap_wall_idx
        wall2 = walls[w2]
        part.pos_loc[p, wall1.wrap_dim] *= -1.0
        part.pw_mask[p,w1] = False
        part.pw_mask[p,w2] = True
        

class PW_WrapLaw(PW_PeriodicLaw):
    name = 'PW_WrapLaw'
    def resolve_collision(self, part, walls, p, w):
        super().resolve_collision(self, part, walls, p, w)
        part.pos = part.pos_loc
        
        
        
class PW_IgnoreLaw(PW_CollisionLaw):
    name = 'PW_IgnoreLaw'
    def resolve_collision(self, part, walls, p, w):
        pass


class PW_TerminateLaw(PW_CollisionLaw):
    name = 'PW_TerminateLaw'
    def resolve_collision(self, part, walls, p, w):
        raise Exception('particle {} hit termination wall {}'.format(p, w))

        
class PW_NoSlipLaw(PW_CollisionLaw):
    name = 'PW_NoSlipLaw'
    def resolve_collision(self, part, walls, p, w):
        nu = walls[w].normal(part.pos_loc[p])
        m = part.mass[p]
        g = part.gamma[p]
        r = part.radius[p]
        d = (2*m*g**2)/(1+g**2)
        
        U_in = part.spin[p]
        v_in = part.vel[p]
        U_out = U_in - (d/(m*g**2) * Lambda_nu(U_in, nu)) + (d/(m*r*g**2)) * E_nu(v_in, nu)
        v_out = (r*d/m) * Gamma_nu(U_in, nu) + v_in - 2 * Pi_nu(v_in, nu) - (d/m) * Pi(v_in,nu)

        part.spin[p] = U_out
        part.vel[p] = v_out