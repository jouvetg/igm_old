#!/usr/bin/env python3

import tensorflow as tf
import math
from igm import igm 

class igm(igm):

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

igm = igm()

igm.config.tstart            = 0
igm.config.tend              = 500
igm.config.tsave             = 2
igm.config.cfl               = 0.3
igm.config.init_strflowctrl  = 90
igm.config.model_lib_path    = '../../model-lib/f12_cfsflow_GJ_21_a'
igm.config.type_mass_balance = 'sinus'
igm.config.usegpu            = True

# Options is for a demo with live plot
#igm.config.plot_live         = True
#igm.config.plot_result       = True
#igm.config.tsave             = 10
#igm.config.varplot_max       = 250

igm.run()
