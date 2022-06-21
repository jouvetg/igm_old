#!/usr/bin/env python3

"""
@author: Guillaume Jouvet
 
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

from igm import Igm
from igm_clim_aletsch import *
from igm_smb_accmelt import *
 
# add extensions for climate generation and mass balance to the core igm class
class Igm(Igm,Igm_clim_aletsch, Igm_smb_accmelt):
    pass

# add custum seeding function, to seeding in the "seeding" area defined in geology.nc
class Igm(Igm):
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
glacier = Igm()

# change parameters
glacier.config.working_dir           = ''
glacier.config.tstart                = 1880
glacier.config.tend                  = 2100
glacier.config.tsave                 = 1
glacier.config.cfl                   = 0.25

glacier.config.iceflow_model_lib_path= '../../model-lib/f15_cfsflow_GJ_22_a'
glacier.config.type_climate          = 'aletsch'

glacier.config.init_slidingco        = 0
glacier.config.init_arrhenius        = 78

# option 1: traditional ams model (acc / melt) -- uncoment these lines
glacier.config.type_mass_balance     = 'accmelt'
glacier.config.massbalance_file      = 'massbalance.nc'

# option 2: emulated smb model by a CNN -- uncoment these lines
#glacier.config.smb_model_lib_path    = '../../model-lib/smb1_meteoswissglamos_GJ_21_a'
#glacier.config.type_mass_balance     = 'nn'
#glacier.config.clim_time_resolution  = 12

glacier.config.weight_accumulation   = 1.00
glacier.config.weight_ablation       = 1.25

glacier.config.usegpu                = True

glacier.config.varplot_max           = 250
glacier.config.plot_result           = False
glacier.config.plot_live             = False

# This permits to give sme weight to accumaulation bassins
glacier.config.weight_Aletschfirn    = 1.0
glacier.config.weight_Jungfraufirn   = 1.0
glacier.config.weight_Ewigschneefeld = 1.0

# This permits to compute particle trajectories
glacier.config.tracking_particles      = False  # activate particle tracking
glacier.config.frequency_seeding       = 10    # we seed every 10 years
glacier.config.density_seeding         = 0.2   # we seed each 5 point of the 2D grid

# From now, we could have call glacier.run(), but we instead give all steps to embed some 
# features like defining initial surface, or check modelled vs observed top DEM std

glacier.initialize()

with tf.device(glacier.device_name):

    glacier.load_ncdf_data(glacier.config.geology_file)
    
    # load the surface toporgaphy available at given year
    glacier.usurf.assign(vars(glacier)['surf_'+str(int(glacier.t))])
    glacier.thk.assign(glacier.usurf-glacier.topg)
    
    glacier.initialize_fields()
    
    while glacier.t < glacier.config.tend:
        
        glacier.tcomp["All"].append(time.time())
             
        # For thes year, check the std between modelled and observed surfaces
        if glacier.t in [1880,1926,1957,1980,1999,2009,2017]:
            diff = (glacier.usurf-vars(glacier)['surf_'+str(int(glacier.t))]).numpy()
            diff = diff[glacier.thk>1]
            mean  = np.mean(diff)
            std   = np.std(diff)
            vol   = np.sum(glacier.thk) * (glacier.dx ** 2) / 10 ** 9
            print(" Check modelled vs observed surface at time : %8.0f ; Mean discr. : %8.2f  ;  Std : %8.2f |  Ice volume : %8.2f " \
                  % (glacier.t, mean, std, vol) )
                  
        glacier.update_climate()
        glacier.update_smb() 
        glacier.update_iceflow()
        if glacier.config.tracking_particles:
            glacier.update_tracking_particles()
            glacier.update_write_trajectories()
        glacier.update_t_dt()
        glacier.update_thk()
        glacier.update_ncdf_ex()
        glacier.update_ncdf_ts()
        glacier.update_plot()
        glacier.print_info()
        
        glacier.tcomp["All"][-1] -= time.time()
        glacier.tcomp["All"][-1] *= -1
        
    glacier.print_all_comp_info()


# Uncomment this to watch the density/weight of debris
# fig, ax = plt.subplots(1,1,figsize=(8,8),dpi=200) 
# im= plt.imshow(igm.weight_particles,origin='lower',vmax=5);  
# cbar = fig.colorbar(im,  orientation="vertical")
# cbar.set_label(label='Density of debris' ,fontsize=12)
# cbar.ax.tick_params(labelsize=12)
# ax.axis('off')
