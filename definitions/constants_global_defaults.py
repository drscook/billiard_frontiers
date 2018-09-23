##################################################################
### Constants ###
##################################################################

BOLTZ = 1.0
np_dtype = np.float64
record_dtype = np.float16
thresh = 1e-10
table_filters = tables.Filters(complevel=1, complib='blosc:lz4')

##################################################################
### Global Variables Defaults ###
##################################################################

check_gpu_cpu = False
same_initial_speeds = True