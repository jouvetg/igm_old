#!/usr/bin/env python3

"""
@author: Guillaume Jouvet
 
"""

import numpy as np
import os, sys, shutil
import os.path
import matplotlib.pyplot as plt
import datetime, time
import glob
import math
import tensorflow as tf
from numpy import dtype
import argparse
from netCDF4 import Dataset
from scipy.interpolate import interp1d

class igm_smb_accmelt:

    def read_config_param_accmelt(self):

        self.parser.add_argument(
            "--weight_ablation", 
            type=float, 
            default=1.0, 
            help="Weight for melt",
        )
        self.parser.add_argument(
            "--weight_accumulation",
            type=float,
            default=1.0,
            help="Weight for accumulation",
        )
        self.parser.add_argument(
            "--thr_temp_snow",
            type=float,
            default=0.5,
            help="Threshold temperature for solid precipitation (0.0)",
        )
        self.parser.add_argument(
            "--thr_temp_rain",
            type=float,
            default=2.5,
            help="Threshold temperature for liquid precipitation (2.0)",
        )

    def init_smb_accmelt(self):
        """
            load smb data to run the Aletsch Glacier simulation 
        """

        nc = Dataset(
            os.path.join(self.config.working_dir, self.config.climate_file), "r"
        )
        x = np.squeeze(nc.variables["x"]).astype("float32")
        y = np.squeeze(nc.variables["y"]).astype("float32")
        self.snow_redistribution = np.squeeze(
            nc.variables["snow_redistribution"]
        ).astype("float32")
        self.direct_radiation = np.squeeze(nc.variables["direct_radiation"]).astype(
            "float32"
        )
        nc.close()

        if not hasattr(self, "x"):
            self.x = tf.constant(x, dtype="float32")
            self.y = tf.constant(y, dtype="float32")
        else:
            assert x.shape == self.x.shape
            assert y.shape == self.y.shape

        # resample at daily resolution the direct radiation field, 
        da = np.arange(20, 351, 30)
        db = np.arange(1, 366)
        self.direct_radiation = interp1d(
            da, self.direct_radiation, kind="linear", axis=0, fill_value="extrapolate")(db)
        
        # shift to hydrological year (i.e. start Oct 1)
        self.direct_radiation = np.roll(self.direct_radiation, 91, axis=0)
        
        self.direct_radiation = tf.Variable(self.direct_radiation, dtype="float32")

        # read mass balance parameters
        self.mb_parameters = tf.Variable(
            np.loadtxt(
                os.path.join(self.config.working_dir, "mbparameter.dat"),
                skiprows=2,
                dtype=np.float32,
            )
        )

        # snow_redistribution must have shape like precipitation (365,ny,nx)
        self.snow_redistribution = tf.expand_dims(self.snow_redistribution, axis=0)
        self.snow_redistribution = tf.tile(self.snow_redistribution, (365, 1, 1))

        # define the year-corresponding indice in the mb_parameters file
        self.IMB = np.zeros((221), dtype="int32")
        for tt in range(1880, 2101):
            self.IMB[tt - 1880] = np.argwhere(
                (self.mb_parameters[:, 0] <= tt) & (tt < self.mb_parameters[:, 1])
            )[0]
        self.IMB = tf.Variable(self.IMB)

    # Warning: The decorator permits to take full benefit from efficient TensorFlow operation (especially on GPU)
    # Note that tf.function works best with TensorFlow ops; NumPy and Python calls are converted to constants.
    # Therefore: you must make sure any variables are TensorFlow Tensor (and not Numpy)
    @tf.function()
    def update_smb_accmelt(self):
        """
            Mass balance forced by climate with accumulation and temperature-index melt model [Hock, 1999; Huss et al., 2009]. 
            It is a TensorFlow re-implementation of the model used in simulations reported in
            (Jouvet and al., Numerical simulation of Rhone's glacier from 1874 to 2100, JCP, 2009)
            (Jouvet and al., Modelling the retreat of Grosser Aletschgletscher in a changing climate, JOG, 2011)
            (Jouvet and Huss, Future retreat of great Aletsch glacier, JOG, 2019)
            (Jouvet and al., Mapping the age of ice of Gauligletscher ..., TC, 2020)
            Check at this latest paper for the model description corresponding to this implementation
        """

        # update melt paramters
        Fm = self.mb_parameters[self.IMB[int(self.t) - 1880], 2] * 10 ** (-3)
        ri = self.mb_parameters[self.IMB[int(self.t) - 1880], 3] * 10 ** (-5)
        rs = self.mb_parameters[self.IMB[int(self.t) - 1880], 4] * 10 ** (-5)

        # keep solid precipitation when temperature < thr_temp_snow
        # with linear transition to 0 between thr_temp_snow and thr_temp_rain
        accumulation = tf.where(
            self.air_temp <= self.config.thr_temp_snow,
            self.precipitation,
            tf.where(
                self.air_temp >= self.config.thr_temp_rain,
                0.0,
                self.precipitation
                * (self.config.thr_temp_rain - self.air_temp)
                / (self.config.thr_temp_rain - self.config.thr_temp_snow),
            ),
        )

        # correct for snow re-distribution
        accumulation *= self.snow_redistribution  # unit to [ m ice eq. / d ]

        accumulation *= self.config.weight_accumulation  # unit to [ m ice eq. / d ]

        pos_temp = tf.where(self.air_temp > 0.0, self.air_temp, 0.0)  # unit is [°C]

        ablation = []  # [ unit : ice-eq. m ]
 
        # the snow depth (=0 or >0) is necessary to find what melt factor to apply
        snow_depth = tf.zeros((self.air_temp.shape[1], self.air_temp.shape[2]))

        for k in range(self.air_temp.shape[0]):

            # add accumulation to the snow depth
            snow_depth += accumulation[k]

            # the ablation (unit is m ice eq. / d) is the product
            # of positive temp  with melt factors for ice, or snow
            # (Fm + ris * self.direct_radiation has unit [m ice eq. / (°C d)] )
            ablation.append(
                tf.where(
                    snow_depth == 0,
                    pos_temp[k] * (Fm + ri * self.direct_radiation[k]),
                    pos_temp[k] * (Fm + rs * self.direct_radiation[k]),
                )
            )

            ablation[-1] *= self.config.weight_ablation

            # remove snow melt to snow depth, and cap it as snow_depth can not be negative
            snow_depth = tf.clip_by_value(snow_depth - ablation[k], 0.0, 1.0e10)

        # Time integration of accumulation minus ablation
        self.smb.assign(tf.math.reduce_sum(accumulation - ablation, axis=0))
        
