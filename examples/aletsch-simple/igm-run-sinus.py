#!/usr/bin/env python3

import tensorflow as tf
import math
from igm import Igm 

class Igm(Igm):

    def update_smb_sinus(self):
        """
            mass balance 'sinus'
        """

        ela     = 2800 + 500*math.sin((self.t/50)*math.pi)
        gradabl = 0.009
        gradacc = 0.005
        maxacc  = 2.0

        smb = self.usurf - ela
        smb *= tf.where(tf.less(smb, 0), gradabl, gradacc)
        smb = tf.clip_by_value(smb, -100, maxacc)
        smb = tf.where(self.icemask>0.5, smb, -10 )

        self.smb.assign( smb )

glacier = Igm()

glacier.config.tstart            = 0
glacier.config.tend              = 500
glacier.config.tsave             = 2
glacier.config.cfl               = 0.3
glacier.config.init_slidingco    = 12
glacier.config.init_arrhenius    = 78
glacier.config.iceflow_model_lib_path    = '../../model-lib/f15_cfsflow_GJ_22_a'
glacier.config.type_mass_balance = 'sinus'
glacier.config.usegpu            = True

# Options is for a demo with live plot
#glacier.config.plot_live         = True
#glacier.config.plot_result       = True
#glacier.config.tsave             = 10
#glacier.config.varplot_max       = 250 

glacier.initialize() 
with tf.device(glacier.device_name):
    glacier.load_ncdf_data(glacier.config.geology_file)
    glacier.initialize_fields()               
    while glacier.t < glacier.config.tend:                       
        glacier.update_smb()
        glacier.update_iceflow()
        glacier.update_t_dt() 
        glacier.update_thk()       
        glacier.update_ncdf_ex()
        glacier.update_ncdf_ts()
        glacier.update_plot()
        glacier.print_info()
        
glacier.print_all_comp_info()
