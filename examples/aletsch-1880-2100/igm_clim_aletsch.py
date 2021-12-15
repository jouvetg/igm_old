#!/usr/bin/env python3

"""
@author: Guillaume Jouvet
 
"""

import os, sys, shutil
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
from igm import igm
import time

class igm_clim_aletsch:

    def read_config_param_climate_aletsch(self):

        # CLIMATE PARAMETERS
        self.parser.add_argument(
            "--clim_time_resolution",
            type=float,
            default=365,
            help="Give the resolution the climate forcing should be (monthly=12, daily=365)",
        ) 

    def load_climate_data_aletsch(self):
        """
            load climate data to run the Aletsch Glacier simulation
        """
        
        # altitude of the weather station for climate data
        self.zws = 2766

        # read temperature and precipitation data from temp_prec.dat
        temp_prec = np.loadtxt(
            os.path.join(self.config.working_dir, "temp_prec.dat"),
            dtype=np.float32,
            skiprows=2,
        )

        # find the min and max years from data
        ymin = int(min(temp_prec[:, 0]))
        ymax = int(max(temp_prec[:, 0]))

        self.temp = np.zeros((365, ymax - ymin + 1), dtype=np.float32)
        self.prec = np.zeros((365, ymax - ymin + 1), dtype=np.float32)
        self.year = np.zeros((ymax - ymin + 1), dtype=np.float32)

        # retrieve temp [unit °C] and prec [unit is m ice eq. / y] and year
        for k, y in enumerate(range(ymin, ymax + 1)):
            IND = (temp_prec[:, 0] == y) & (temp_prec[:, 1] <= 365)
            self.prec[:, k] = (
                temp_prec[IND, -1] * 365.0 / 1000.0
            ) / 0.917  # new unit is m ice eq. / y
            self.temp[:, k] = temp_prec[IND, -2]  # new unit is °C
            self.year[k] = y

        # this make monthly temp and prec if this is wished
        if self.config.clim_time_resolution==12:
            II = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 364]
            self.prec = np.stack([np.mean(self.prec[II[i]:II[i+1]],axis=0) 
                                  for i in range(0,12) ] )
            self.temp = np.stack([np.mean(self.temp[II[i]:II[i+1]],axis=0) 
                                  for i in range(0,12) ] )

        # intitalize air_temp and precipitation fields
        self.air_temp = tf.Variable(
            tf.zeros((len(self.temp), self.y.shape[0], self.x.shape[0])),
            dtype="float32",
        )
        self.precipitation = tf.Variable(
            tf.zeros((len(self.prec), self.y.shape[0], self.x.shape[0])),
            dtype="float32",
        )

    def update_climate_aletsch(self, force=False):

        dP = 0.00035   # Precipitation vertical gradient
        dT = -0.00552  # Temperature vertical gradient

        # find out the precipitation and temperature at the weather station
        II = self.year == int(self.t)
        PREC = tf.expand_dims(
            tf.expand_dims(np.squeeze(self.prec[:, II]), axis=-1), axis=-1
        )
        TEMP = tf.expand_dims(
            tf.expand_dims(np.squeeze(self.temp[:, II]), axis=-1), axis=-1
        )

        # extend air_temp and precipitation over the entire glacier and all day of the year
        self.precipitation.assign(tf.tile(PREC, (1, self.y.shape[0], self.x.shape[0])))
        self.air_temp.assign(tf.tile(TEMP, (1, self.y.shape[0], self.x.shape[0])))

        # vertical correction (lapse rates)
        prec_corr_mult = 1 + dP * (self.usurf - self.zws)
        temp_corr_addi = dT * (self.usurf - self.zws)

        prec_corr_mult = tf.expand_dims(prec_corr_mult, axis=0)
        temp_corr_addi = tf.expand_dims(temp_corr_addi, axis=0)

        prec_corr_mult = tf.tile(prec_corr_mult, (len(self.prec), 1, 1))
        temp_corr_addi = tf.tile(temp_corr_addi, (len(self.temp), 1, 1))

        # the final precipitation and temperature must have shape (365,ny,ny)
        self.precipitation.assign(tf.clip_by_value(self.precipitation * prec_corr_mult, 0, 10 ** 10))
        self.air_temp.assign(self.air_temp + temp_corr_addi)
        
