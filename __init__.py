import billiards.util

import os
from copy import copy, deepcopy
import itertools as it
import math
import IPython.display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from billiards.util import utility_functions as ut


from billiards.util.display_preferences import *
plt.style.use("fivethirtyeight")

# import tables
# from billiards.dynamics.no_slip_functions import *
# from billiards.dynamics.pw_collision_laws import *
# from billiards.dynamics.pp_collision_laws import *
# from billiards.dynamics.particles import *
# from billiards.dynamics.walls import *
# from billiards.dynamics.dynamics import *