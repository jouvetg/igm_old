#!/usr/bin/env python3

import tensorflow as tf
import math
from igm import igm 

igm = igm()

igm.config.tstart            = 1000
igm.config.tend              = 1250
igm.config.tsave             = 2
igm.config.cfl               = 0.5
igm.config.init_strflowctrl  = 90
igm.config.iceflow_model_lib_path    = '../../model-lib/f12_cfsflow_GJ_21_a'
igm.config.type_mass_balance = 'simple'
igm.config.usegpu            = True

# Options is for a demo with live plot
# igm.config.plot_live         = True
# igm.config.plot_result       = True
# igm.config.tsave             = 10
# igm.config.varplot_max       = 250

igm.run()
