#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
 
from igm import * 

class Igm(Igm):
    pass

glacier = Igm() 

glacier.config.version = 'v2' 

glacier.config.iceflow_model_lib_path= '../../model-lib/f21_pinnbp_GJ_23_a'
 
glacier.config.plot_result           = True
glacier.config.plot_live             = True
glacier.config.observation_file      = 'observation.nc' # this is the main input file
 
glacier.config.opti_output_freq              = 100    # Frequency for output
glacier.config.opti_nbitmax                  = 2000   # Number of it. for the optimization

glacier.config.opti_regu_param_thk           = 2 # weight for the regul. of thk

glacier.config.opti_step_size_v2             = 0.25

glacier.config.opti_smooth_anisotropy_factor = 0.2   # Smooth anisotropy factor

glacier.config.opti_usurfobs_std             = 5.0   # Tol to fit top ice surface
glacier.config.opti_velsurfobs_std           = 2.0   # Tol to fit surface speeds
glacier.config.opti_thkobs_std               = 5.0   # Tol to fit ice thk profiles
glacier.config.opti_divfluxobs_std           = 1.0   # Tol to fit top ice surface

glacier.config.opti_regu_param_slidingco  = 10    # weight for the regul. of slidingco
glacier.config.opti_slidingco_std         = 5000.0   # Tol to fit strflowctr
glacier.config.opti_thr_slidingco         = 78000

glacier.config.init_slidingco = 10000
 
glacier.config.opti_control=['thk','slidingco']
glacier.config.opti_cost=['velsurf','icemask','thk'] 
 
glacier.initialize()

with tf.device(glacier.device_name): 

    glacier.load_ncdf_data(glacier.config.observation_file)
    glacier.initialize_fields()   
    glacier.update_iceflow_emulated() # used to load the ML ice flow emaultor
    glacier.optimize()

glacier.print_all_comp_info()
     
 