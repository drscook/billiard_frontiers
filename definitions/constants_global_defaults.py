##################################################################
### Constants ###
##################################################################

BOLTZ = 1.0
np_dtype = np.float64
thresh = 1e-10
table_filters = tables.Filters(complevel=1, complib='blosc:lz4')

##################################################################
### Global Variables Defaults ###
##################################################################

max_steps = 100
mode = 'serial'
check_gpu_cpu = False
same_initial_speeds = True


dim = 2
part_num = 20
part_mass = 1.0
part_radius = 1.0
part_gamma = 'uniform'
part_temp = 1.0
record_vars = ['t', 'pos', 'vel', 'spin']#, 'col']