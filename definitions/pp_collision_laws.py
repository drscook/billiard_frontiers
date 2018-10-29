class PP_CollisionLaw:
    @staticmethod
    def resolve_collision(self, part, p1, p2):
        raise Exception('You should implement the method resolve_collision() in a subclass.')

class PP_IgnoreLaw(PP_CollisionLaw):
    name = 'PP_IgnoreLaw'
    def resolve_collision(self, part, p1, p2):
        pass

class PP_SpecularLaw(PP_CollisionLaw):
    name = 'PP_SpecularLaw'        
    def resolve_collision(self, part, p1, p2):
        nu = part.pos[p2] - part.pos[p1]
        nu = make_unit(nu)
        m1 = part.mass[p1]
        m2 = part.mass[p2]
        M = m1 + m2

        dv = part.vel[p2] - part.vel[p1]
        w = dv.dot(nu) * nu
        part.vel[p1] += 2 * (m2/M) * w
        part.vel[p2] -= 2 * (m1/M) * w

class PP_NoSlipLaw(PP_CollisionLaw):
    name = 'PP_NoSlipLaw'
    def resolve_collision(self, part, p1, p2):
        print("Using PP_NoSlipLaw")
        nu = part.pos[p2] - part.pos[p1]
        nu = make_unit(nu)
        m1 = part.mass[p1]
        m2 = part.mass[p2]
        M = m1 + m2
        
        r1 = part.radius[p1]
        r2 = part.radius[p2]        
        g1 = part.gamma[p1]
        g2 = part.gamma[p2]
        d = 2/((1/m1)*(1+1/g1**2) + (1/m2)*(1+1/g2**2))
        
        U1_in = part.spin[p1]
        U2_in = part.spin[p2]
        v1_in = part.vel[p1]
        v2_in = part.vel[p2]

        U1_out = (U1_in-d/(m1*g1**2) * Lambda_nu(U1_in, nu)) \
                    + (-d/(m1*r1*g1**2)) * E_nu(v1_in, nu) \
                    + (-r2/r1)*(d/(m1*g1**2)) * Lambda_nu(U2_in, nu) \
                    + d/(m1*r1*g1**2) * E_nu(v2_in, nu)

        v1_out = (-r1*d/m1) * Gamma_nu(U1_in, nu) \
                    + (v1_in - 2*m2/M * Pi_nu(v1_in, nu) - (d/m1) * Pi(v1_in, nu)) \
                    + (-r2*d/m1) * Gamma_nu(U2_in, nu) \
                    + (2*m2/M) * Pi_nu(v2_in, nu) + (d/m1) * Pi(v2_in, nu)

        U2_out = (-r1/r2)*(d/(m2*g2**2)) * Lambda_nu(U1_in, nu) \
                    + (-d/(m2*r2*g2**2)) * E_nu(v1_in, nu) \
                    + (U2_in - (d/(m2*g2**2)) * Lambda_nu(U2_in, nu)) \
                    + (d/(m2*r2*g2**2)) * E_nu(v2_in, nu)

        v2_out = (r1*d/m2) * Gamma_nu(U1_in, nu) \
                    + (2*m1/M) * Pi_nu(v1_in, nu) + (d/m2) * Pi(v1_in, nu) \
                    + (r2*d/m2) * Gamma_nu(U2_in, nu) \
                    + v2_in - (2*m1/M) * Pi_nu(v2_in, nu) - (d/m2) * Pi(v2_in,nu)
        part.spin[p1] = U1_out
        part.spin[p2] = U2_out
        part.vel[p1] = v1_out
        part.vel[p2] = v2_out   