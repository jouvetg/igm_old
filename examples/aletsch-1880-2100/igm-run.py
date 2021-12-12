#!/usr/bin/env python3

"""
@author: Guillaume Jouvet
 
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

from igm import *
from igm_clim_aletsch import *
from igm_smb_accmelt import *
 
# add extensions for climate generation and mass balance to the core igm class
class igm(igm,igm_clim_aletsch, igm_smb_accmelt):
    pass

# define the igm class
igm = igm()

# change parameters
igm.config.working_dir           = '' 
igm.config.tstart                = 1880
igm.config.tend                  = 2100
igm.config.tsave                 = 1
igm.config.init_strflowctrl      = 78
igm.config.cfl                   = 0.25

igm.config.model_lib_path        = '../../model-lib/f12_cfsflow_GJ_21_a'
igm.config.type_mass_balance     = 'accmelt'
igm.config.type_climate          = 'aletsch'
igm.config.climate_file          = 'massbalance.nc'

igm.config.weight_accumulation   = 1.00
igm.config.weight_ablation       = 1.25

igm.config.usegpu                = True

igm.config.varplot_max           = 250
igm.config.plot_result           = False
igm.config.plot_live             = False

# From now, we could have call igm.run(), but we instead give all steps to embed some 
# features like defining initial surface, or check modelled vs observed top DEM std

igm.initialize()

with tf.device(igm.device_name):

    igm.load_ncdf_data(igm.config.geology_file)
    
    # load the surface toporgaphy available at given year
    igm.usurf.assign(vars(igm)['surf_'+str(int(igm.t))])
    igm.thk.assign(igm.usurf-igm.topg)
    
    igm.initialize_fields()
    igm.initialize_iceflow()
    igm.update_climate()
    igm.update_smb()
    igm.update_iceflow()
    igm.update_ncdf_ex()
    igm.update_ncdf_ts()
    igm.print_info()

    while igm.t < igm.config.tend:
             
        # For thes year, check the std between modelled and observed surfaces
        if igm.t in [1926,1957,1980,1999,2009,2017]:
            diff = (igm.usurf-vars(igm)['surf_'+str(int(igm.t))]).numpy()
            diff = diff[igm.thk>1]
            mean  = np.mean(diff)
            std   = np.std(diff)
            vol   = np.sum(igm.thk) * (igm.dx ** 2) / 10 ** 9
            print(" Check modelled vs observed surface at time : %8.0f ; Mean discr. : %8.2f  ;  Std : %8.2f |  Ice volume : %8.2f " \
                  % (igm.t, mean, std, vol) )
        
        igm.update_climate()
        igm.update_smb() 
        igm.update_iceflow()
        igm.update_t_dt()
        igm.update_thk()
        igm.update_ncdf_ex()
        igm.update_ncdf_ts()
        igm.update_plot()
        igm.print_info()
        
    igm.print_all_comp_info()
