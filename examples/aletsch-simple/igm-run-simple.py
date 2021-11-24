#!/usr/bin/env python3

import tensorflow as tf
from igm import igm

igm = igm()

igm.config.tstart            = 1000
igm.config.tend              = 1250
igm.config.tsave             = 2
igm.config.cfl               = 0.3
igm.config.init_strflowctrl  = 90
igm.config.model_lib_path    = '../../model-lib/f12_cfsflow_GJ_21_a'
igm.config.type_mass_balance = 'simple'
igm.config.usegpu            = True

# Options is for a demo with live plot
# igm.config.plot_live         = True
# igm.config.plot_result       = True
# igm.config.tsave             = 10
# igm.config.varplot_max       = 250

igm.run()

# THE FOLLING CAN BE REPLACE BY : igm.run()

# igm.initialize()

# with tf.device(igm.device_name):

#     igm.load_ncdf_data(igm.config.geology_file)
#     igm.initialize_fields()
#     igm.initialize_iceflow()
#     igm.update_smb()
#     igm.update_iceflow()
#     igm.update_ncdf_ex()
#     igm.print_info()

#     while igm.t < igm.config.tend:
#         igm.update_smb()
#         igm.update_iceflow()
#         igm.update_t_dt()
#         igm.update_thk()
#         igm.update_ncdf_ex()
#         igm.update_plot()
#         igm.print_info()

