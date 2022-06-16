#!/usr/bin/env python3

"""
@author: Guillaume Jouvet
 
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

import igm
from igm_clim_aletsch import *
from igm_smb_accmelt import *
 
# add extensions for climate generation and mass balance to the core igm class
class igm(igm,igm_clim_aletsch, igm_smb_accmelt):
    pass

# add custum seeding function, to seeding in the "seeding" area defined in geology.nc
class igm(igm):
    def seeding_particles(self):
        # here we seed where i) thickness is higher than 1 m
        #                    ii) the seeding field of geology.nc is active
        #                    iii) on the gridseed (which permit to control the seeding density)
        I = (self.thk>1)&(self.seeding>0.5)&self.gridseed  # here you may redefine how you want to seed particles
        self.nxpos  = self.X[I]                # x position of the particle
        self.nypos  = self.Y[I]                # y position of the particle
        self.nrhpos = tf.ones_like(self.X[I])  # relative position in the ice column
        self.nwpos  = tf.ones_like(self.X[I])  # this is the weight of the particle

# define the igm class
igm = igm()

# change parameters
igm.config.working_dir           = ''
igm.config.tstart                = 1880
igm.config.tend                  = 2100
igm.config.tsave                 = 1
igm.config.cfl                   = 0.25

igm.config.iceflow_model_lib_path= '../../model-lib/f15_cfsflow_GJ_22_a'
igm.config.type_climate          = 'aletsch'

# option 1: traditional ams model (acc / melt) -- uncoment these lines
igm.config.type_mass_balance     = 'accmelt'
igm.config.massbalance_file      = 'massbalance.nc'

# option 2: emulated smb model by a CNN -- uncoment these lines
#igm.config.smb_model_lib_path    = '../../model-lib/smb1_meteoswissglamos_GJ_21_a'
#igm.config.type_mass_balance     = 'nn'
#igm.config.clim_time_resolution  = 12

igm.config.weight_accumulation   = 1.00
igm.config.weight_ablation       = 1.25

igm.config.usegpu                = True

igm.config.varplot_max           = 250
igm.config.plot_result           = True
igm.config.plot_live             = True

# This permits to give sme weight to accumaulation bassins
igm.config.weight_Aletschfirn    = 1.0
igm.config.weight_Jungfraufirn   = 1.0
igm.config.weight_Ewigschneefeld = 1.0

# This permits to compute particle trajectories
igm.config.tracking_particles      = True  # activate particle tracking
igm.config.frequency_seeding       = 2    # we seed every 10 years
igm.config.density_seeding         = 0.2   # we seed each 5 point of the 2D grid

# From now, we could have call igm.run(), but we instead give all steps to embed some 
# features like defining initial surface, or check modelled vs observed top DEM std

igm.initialize()

with tf.device(igm.device_name):

    igm.load_ncdf_data(igm.config.geology_file)
    
    # load the surface toporgaphy available at given year
    igm.usurf.assign(vars(igm)['surf_'+str(int(igm.t))])
    igm.thk.assign(igm.usurf-igm.topg)
    
    igm.initialize_fields()
    
    while igm.t < igm.config.tend:
        
        igm.tcomp["All"].append(time.time())
             
        # For thes year, check the std between modelled and observed surfaces
        if igm.t in [1880,1926,1957,1980,1999,2009,2017]:
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
        if igm.config.tracking_particles:
            igm.update_tracking_particles()
            igm.update_write_trajectories()
        igm.update_t_dt()
        igm.update_thk()
        igm.update_ncdf_ex()
        igm.update_ncdf_ts()
        igm.update_plot()
        igm.print_info()
        
        igm.tcomp["All"][-1] -= time.time()
        igm.tcomp["All"][-1] *= -1
        
    igm.print_all_comp_info()


# Uncomment this to watch the density/weight of debris
# fig, ax = plt.subplots(1,1,figsize=(8,8),dpi=200) 
# im= plt.imshow(igm.weight_particles,origin='lower',vmax=5);  
# cbar = fig.colorbar(im,  orientation="vertical")
# cbar.set_label(label='Density of debris' ,fontsize=12)
# cbar.ax.tick_params(labelsize=12)
# ax.axis('off')