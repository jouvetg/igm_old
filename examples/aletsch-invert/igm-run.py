#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from igm import igm

igm = igm() 

igm.config.working_dir           = '' 
#igm.config.iceflow_model_lib_path= '../../model-lib/f12_cfsflow'
igm.config.iceflow_model_lib_path='../../model-lib/f17_pismbp_GJ_22_a'
 
igm.config.plot_result           = True
igm.config.plot_live             = True
igm.config.observation_file      = 'observation.nc' # this is the main input file
 
igm.config.opti_output_freq              = 50     # Frequency for output
igm.config.opti_nbitmax                  = 1000   # Number of it. for the optimization

igm.config.opti_regu_param_thk           = 10.0   # weight for the regul. of thk
igm.config.opti_regu_param_strflowctrl   = 1.0    # weight for the regul. of strflowctrl

igm.config.opti_step_size                = 0.001

igm.config.opti_smooth_anisotropy_factor = 0.2   # Smooth anisotropy factor

igm.config.opti_usurfobs_std             = 5.0   # Tol to fit top ice surface
igm.config.opti_velsurfobs_std           = 3.0   # Tol to fit surface speeds
igm.config.opti_thkobs_std               = 5.0   # Tol to fit ice thk profiles
igm.config.opti_divfluxobs_std           = 1.0   # Tol to fit top ice surface

# Uncomment for Optimization scheme O
igm.config.opti_control=['thk','strflowctrl','usurf']
igm.config.opti_cost=['velsurf','thk','usurf','divfluxfcz','icemask']  

# Uncomment for Optimization scheme $O_{-\tilde{A}}$
#igm.config.opti_control=['thk','usurf']
#igm.config.opti_cost=['velsurf','thk','usurf','divfluxfcz','icemask']  

# Uncomment for Optimization scheme $O_{-\tilde{A},h}$
#igm.config.opti_control=['thk','usurf']
#igm.config.opti_cost=['velsurf','usurf','divfluxfcz','icemask']  

#Uncomment for Optimization scheme $O_{-d}$
#igm.config.opti_control=['thk','strflowctrl','usurf']
#igm.config.opti_cost=['velsurf','thk','usurf','icemask']  

# Uncomment for Optimization scheme $O_{-s}$
#igm.config.opti_control=['thk','strflowctrl']
#igm.config.opti_cost=['velsurf','thk','divfluxfcz','icemask']  

igm.initialize()

with tf.device(igm.device_name):
    igm.load_ncdf_data(igm.config.observation_file)
    igm.initialize_fields()
    igm.optimize()

igm.print_all_comp_info()

