# import basic libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys 

# import the igm library
from igm import *

# We augment igm with a custumized smb function that takes ELA and gradmb previously defined
class igm(igm):

    def update_smb_mysmb(self):
        """
            mass balance 'mysmb'
        """

        if self.t<1880:
           ela = self.ELA + 0  # Take today's ELA + 50 before 1800
        else:
           ela = self.ELA + 25  # Take today's ELA + 100 after 1800

        smb = (self.usurf - ela)*self.gradmb
        smb = tf.where(self.icemask>0.5, smb, -10 )
        self.smb.assign( smb )
 
# create an object igm, which contains variable and functions
igm = igm() 

# ELA and gradmb values needs to be accessible from the class igm
igm.ELA    = 2900
igm.gradmb = 0.003

igm.config.working_dir              = '' 
igm.config.iceflow_model_lib_path   = '../../model-lib/f17_pismbp_GJ_22_a/'  
igm.config.geology_file             = 'geology.nc' # this time we take the (shortly) optimized geology data

igm.config.tstart            = 1700    # our data (glacier surface DEM, speeds) are about from that year
igm.config.tend              = 2020    # we run the model until 2100
igm.config.tsave             = 2       # we save output each 2 years
igm.config.cfl               = 0.15    # the CFL number determines the time step, it should be < 1, reduce it further if unstability occurs.
igm.config.type_mass_balance = 'mysmb' # we have to select the SMB defined before.

igm.config.plot_result           = True
igm.config.plot_live             = True

igm.config.varplot_max           = 250
igm.config.vel3d_active          = False 

# This permits to compute particle trajectories
igm.config.tracking_particles      = False  # activate particle tracking
igm.config.frequency_seeding       = 20    # we seed every 10 years
igm.config.density_seeding         = 0.2   # we seed each 5 point of the 2D grid

igm.config.vars_to_save= [ "topg", "usurf", "thk", "smb", "velbar_mag", "velsurf_mag", 'uvelsurf', 'vvelsurf' ]
 
igm.run()

igm.animate_result('ex.nc','thk',save=False)
igm.animate_result('ex.nc','velsurf_mag',save=False)

