# import basic libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys 

# import the igm library
from igm import Igm

# We augment igm with a custumized smb function that takes ELA and gradmb previously defined
class Igm(Igm):

    def update_smb_mysmb(self):
        """
            mass balance 'mysmb'
        """

        ela = self.ELA
        smb = (self.usurf - ela)*self.gradmb
        smb = tf.where(self.icemask>0.5, smb, -10 )
        self.smb.assign( smb )
 
# create an object igm, which contains variable and functions
glacier = Igm() 

# ELA and gradmb values needs to be accessible from the class igm
glacier.ELA    = 3200
glacier.gradmb = 0.005

glacier.config.working_dir              = '' 
glacier.config.iceflow_model_lib_path   = '../../model-lib/f15_cfsflow_GJ_22_a/'  
glacier.config.geology_file             = 'geology.nc' # this time we take the (shortly) optimized geology data

glacier.config.tstart            = 2000    # our data (glacier surface DEM, speeds) are about from that year
glacier.config.tend              = 2400    # we run the model until 2100
glacier.config.tsave             = 5       # we save output each 2 years
glacier.config.cfl               = 0.15    # the CFL number determines the time step, it should be < 1, reduce it further if unstability occurs.
glacier.config.type_mass_balance = 'mysmb' # we have to select the SMB defined before.

glacier.config.plot_result           = True
glacier.config.plot_live             = True

glacier.config.varplot_max           = 250 

# This permits to compute particle trajectories 
glacier.config.tracking_method         ='simple'
glacier.config.frequency_seeding       = 1000    # we seed every 10 years
glacier.config.density_seeding         = 0.1   # we seed each 5 point of the 2D grid
 
glacier.initialize() 
with tf.device(glacier.device_name):
    glacier.load_ncdf_data(glacier.config.geology_file)
    glacier.initialize_fields()               
    while glacier.t < glacier.config.tend:                       
        glacier.update_smb()
        glacier.update_iceflow()
        if glacier.t>=2200:
             glacier.update_particles()
        glacier.update_t_dt() 
        glacier.update_thk()       
        glacier.update_ncdf_ex()
        glacier.update_ncdf_ts()
        glacier.update_plot()
        glacier.print_info() 