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
check_gpu_cpu = False
same_initial_speeds = True