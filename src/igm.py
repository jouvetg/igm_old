#!/usr/bin/env python3

"""
Copyright (C) 2021-2022 Guillaume Jouvet <guillaume.jouvet@geo.uzh.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file

This file contains the core functions the Instructed Glacier Model (IGM),
functions are sorted by thema, which are

#      INITIALIZATION : contains all to initialize variables
#      I/O NCDF : Files for data Input/Output using NetCDF file format
#      COMPUTE T AND DT : Function to compute adaptive time-step and time
#      ICEFLOW : Containt the function that serve compute the emulated iceflow
#      CLIMATE : Templates for function updating climate
#      MASS BALANCE : Simple mass balance
#      TRANSPORT : This handles the mass conservation
#      PLOT : Plotting functions
#      PRINT INFO : handle the printing of output during computation
#      RUN : the main function that wrap all functions within an time-iterative loop

"""

####################################################################################

import os, sys, shutil, glob
import numpy as np
import matplotlib.pyplot as plt
import datetime, time
import math
import tensorflow as tf
from netCDF4 import Dataset
from scipy.interpolate import interp1d, CubicSpline, RectBivariateSpline, griddata
import argparse
from IPython.display import display, clear_output
 
def str2bool(v):
    return v.lower() in ("true", "1")

####################################################################################

class igm:

    ####################################################################################
    ####################################################################################
    ####################################################################################
    #                               INITIALIZATION
    ####################################################################################
    ####################################################################################
    ####################################################################################

    def __init__(self):
        """
            function build class IGM
        """
        self.parser = argparse.ArgumentParser(description="IGM")
        self.read_config_param()
        self.config = self.parser.parse_args()

    def read_config_param(self):

        self.parser.add_argument(
            "--working_dir", type=str, default="", help="Working directory"
        )
        self.parser.add_argument(
            "--geology_file", type=str, default="geology.nc", help="Geology input file"
        )
        self.parser.add_argument(
            "--tstart", type=float, default=0.0, help="Starting time"
        )
        self.parser.add_argument("--tend", type=float, default=150.0, help="End time")
        self.parser.add_argument(
            "--restartingfile",
            type=str,
            default="",
            help="Provide restarting file if no empty string",
        )
        self.parser.add_argument(
            "--verbosity",
            type=int,
            default=0,
            help="Verbosity level of IGM (default: 0 = no verbosity)",
        )
        self.parser.add_argument(
            "--tsave", type=float, default=10, help="Save result each X years (10)"
        )
        self.parser.add_argument(
            "--plot_result", type=str2bool, default=False, help="Plot results in png",
        )
        self.parser.add_argument(
            "--plot_live",
            type=str2bool,
            default=False,
            help="Display plots live the results during computation",
        )
        self.parser.add_argument(
            "--usegpu",
            type=str2bool,
            default=True,
            help="Use the GPU for ice flow model (True)",
        )
        self.parser.add_argument(
            "--stop",
            type=str2bool,
            default=False,
            help="experimental, just o get fair comp time, to be removed ....",
        )
        self.parser.add_argument(
            "--init_strflowctrl", type=float, default=78, help="Initial strflowctrl",
        )
        self.parser.add_argument(
            "--optimize",
            type=str2bool,
            default=False,
            help="Optimize prior forward modelling  (not available yet)",
        )
        self.parser.add_argument(
            "--update_topg",
            type=str2bool,
            default=False,
            help="Update bedrock (not available yet)",
        )

        for p in dir(self):
            if "read_config_param_" in p:
                getattr(self, p)()

    ##################################################

    def initialize(self):
        """
            function initialize the strict minimum for IGM class, and record parameters
        """

        print(
            "+++++++++++++++++++ START IGM ++++++++++++++++++++++++++++++++++++++++++"
        )

        self.t = tf.Variable(float(self.config.tstart))
        self.it = 0
        self.dt = float(self.config.dtmax)
        self.dt_target = float(self.config.dtmax)

        self.saveresult = True

        self.tcomp = {}
        self.tcomp["All"] = []

        self.device_name = "/GPU:0" * self.config.usegpu + "/CPU:0" * (
            not self.config.usegpu
        )

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi

        if self.config.usegpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1" for multiple
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # This is used to limite the number of used core to 1 (or user-defined)
        #        tf.config.threading.set_inter_op_parallelism_threads(1)  # another method
        #        tf.config.threading.set_intra_op_parallelism_threads(1)  # another method

        with open(
            os.path.join(self.config.working_dir, "igm-run-parameters.txt"), "w"
        ) as f:
            print("PARAMETERS ARE ...... ")
            for ck in self.config.__dict__:
                print("%30s : %s" % (ck, self.config.__dict__[ck]))
                print("%30s : %s" % (ck, self.config.__dict__[ck]), file=f)

    def load_ncdf_data(self, filename):
        """
            Load the geological input files from netcdf file
        """
        if self.config.verbosity == 1:
            print("LOAD NCDF file")

        nc = Dataset(os.path.join(self.config.working_dir, filename), "r")

        if not hasattr(self, "x"):
            self.x = tf.constant(np.squeeze(nc.variables["x"]).astype("float32"))
            self.y = tf.constant(np.squeeze(nc.variables["y"]).astype("float32"))
            assert self.x[1] - self.x[0] == self.y[1] - self.y[0]
        else:
            xx = tf.constant(np.squeeze(nc.variables["x"]).astype("float32"))
            yy = tf.constant(np.squeeze(nc.variables["y"]).astype("float32"))
            assert self.x == xx
            assert self.y == yy

        for var in nc.variables:
            if not var in ["x", "y"]:
                vars(self)[var] = tf.Variable(
                    np.squeeze(nc.variables[var]).astype("float32")
                )

        nc.close()

    def initialize_fields(self):
        """
            Initialize fields, complete the loading of geology
        """

        if self.config.verbosity == 1:
            print("Initialize fields")

        # at this point, we should have defined at least x, y, usurf
        assert hasattr(self, "x")
        assert hasattr(self, "y")
        assert hasattr(self, "usurf")

        if not hasattr(self, "thk"):
            self.thk = tf.Variable(tf.zeros((self.y.shape[0], self.x.shape[0])))

        self.topg = tf.Variable(self.usurf - self.thk)

        if not hasattr(self, "icemask"):
            if hasattr(self, "mask"):
                self.icemask = tf.Variable(self.mask)
            else:
                self.icemask = tf.Variable(tf.ones_like(self.thk))

        if not hasattr(self, "uvelsurf"):
            self.uvelsurf = tf.Variable(tf.zeros_like(self.thk))

        if not hasattr(self, "vvelsurf"):
            self.vvelsurf = tf.Variable(tf.zeros_like(self.thk))

        if not hasattr(self, "smb"):
            if hasattr(self, "mb"):
                self.smb = tf.Variable(self.mb)
            else:
                self.smb = tf.Variable(tf.zeros_like(self.thk))

        if not hasattr(self, "dhdt"):
            self.dhdt = tf.Variable(tf.zeros_like(self.thk))

        if not hasattr(self, "strflowctrl"):
            self.strflowctrl = tf.Variable(tf.ones_like(self.thk) * self.config.init_strflowctrl)

        self.X, self.Y = tf.meshgrid(self.x, self.y)

        self.dx = self.x[1] - self.x[0]

        self.slopsurfx, self.slopsurfy = self.compute_gradient_tf(
            self.usurf, self.dx, self.dx
        )

        self.ubar = tf.Variable(tf.zeros_like(self.thk))
        self.vbar = tf.Variable(tf.zeros_like(self.thk))
        self.uvelbase = tf.Variable(tf.zeros_like(self.thk))
        self.vvelbase = tf.Variable(tf.zeros_like(self.thk))
        self.divflux = tf.Variable(tf.zeros_like(self.thk))

        self.var_info = {}
        self.var_info["topg"] = ["Basal Topography", "m"]
        self.var_info["usurf"] = ["Surface Topography", "m"]
        self.var_info["thk"] = ["Ice Thickness", "m"]
        self.var_info["icemask"] = ["Ice mask", "NO UNIT"]
        self.var_info["smb"] = ["Surface Mass Balance", "m/y"]
        self.var_info["ubar"] = ["x depth-average velocity of ice", "m/y"]
        self.var_info["vbar"] = ["y depth-average velocity of ice", "m/y"]
        self.var_info["velbar_mag"] = ["Depth-average velocity magnitude of ice", "m/y"]
        self.var_info["uvelsurf"] = ["x surface velocity of ice", "m/y"]
        self.var_info["vvelsurf"] = ["y surface velocity of ice", "m/y"]
        self.var_info["velsurf_mag"] = ["Surface velocity magnitude of ice", "m/y"]
        self.var_info["uvelbase"] = ["x basal velocity of ice", "m/y"]
        self.var_info["vvelbase"] = ["y basal velocity of ice", "m/y"]
        self.var_info["velbase_mag"] = ["Basal velocity magnitude of ice", "m/y"]
        self.var_info["divflux"] = ["Divergence of the ice flux", "m/y"]
        self.var_info["strflowctrl"] = ["arrhenius + 10 * slidingco", "MPa$^{-3}$ a$^{-1}$"]
        self.var_info["arrhenius"] = ["Arrhenius factor", "MPa$^{-3}$ a$^{-1}$"]
        self.var_info["slidingco"] = ["Sliding Coefficient", "km MPa$^{-3}$ a$^{-1}$"]
        self.var_info["meantemp"] = ["Mean anual surface temperatures", "°C"]
        self.var_info["meanprec"] = ["Mean anual precipitation", "m/y"]
        self.var_info["vol"] = ["Ice volume", "km^3"]
        self.var_info["area"] = ["Glaciated area", "km^2"]

        self.var_info["velsurfobs_mag"] = [
            "Observed Surface velocity magnitude of ice",
            "m/y",
        ]

    def restart(self):
        """
            Permit to restart a simulation from a ncdf file y taking the last iterate
        """
        if self.config.verbosity == 1:
            print("READ RESTARTING FILE, OVERIDE FIELDS WHEN GIVEN")

        nc = Dataset(self.config.restartingfile, "r")

        Rx = tf.constant(np.squeeze(nc.variables["x"]).astype("float32"))
        Ry = tf.constant(np.squeeze(nc.variables["y"]).astype("float32"))

        assert Rx.shape == self.x.shape
        assert Ry.shape == self.y.shape

        for var in nc.variables:
            if not var in ["x", "y"]:
                vars(self)[var].assign(
                    np.squeeze(nc.variables[var][-1]).astype("float32")
                )

        nc.close()

    def stop(self):
        """
            this is a dummy stop that serves to syncrohnize CPU and GPU for reliable computational times
        """
        ubar_np = self.ubar.numpy()
        thk_np = self.thk.numpy()
        mb_np = self.smb.numpy()

    def whatsize(self, n):
        s = float(n.size * n.itemsize) / (10 ** 6)
        print("%1.0f Mbytes" % s)

    ####################################################################################
    ####################################################################################
    ####################################################################################
    #                               I/O NCDF
    ####################################################################################
    ####################################################################################
    ####################################################################################

    def read_config_param_output_ncdf(self):

        self.parser.add_argument(
            "--vars_to_save",
            type=list,
            default=[
                "topg",
                "usurf",
                "thk",
                "smb",
                "velbar_mag",
                "velsurf_mag",
            ],
            help="List of variables to be recorded in the ncdef file",
        )

    def update_ncdf_ex(self, force=False):
        """
            Initialize  and write the ncdf output file
        """

        if not hasattr(self, "already_called_update_ncdf_ex"):
            self.tcomp["Outputs ncdf"] = []
            self.already_called_update_ncdf_ex = True

        if force | self.saveresult:

            self.tcomp["Outputs ncdf"].append(time.time())

            if "velbar_mag" in self.config.vars_to_save:
                self.velbar_mag = self.getmag(self.ubar, self.vbar)

            if "velsurf_mag" in self.config.vars_to_save:
                self.velsurf_mag = self.getmag(self.uvelsurf, self.vvelsurf)

            if "velbase_mag" in self.config.vars_to_save:
                self.velbase_mag = self.getmag(self.uvelbase, self.vvelbase)

            if "meanprec" in self.config.vars_to_save:
                self.meanprec = tf.math.reduce_mean(self.precipitation, axis=0)

            if "meantemp" in self.config.vars_to_save:
                self.meantemp = tf.math.reduce_mean(self.air_temp, axis=0)

            if self.it == 0:

                if self.config.verbosity == 1:
                    print("Initialize NCDF output Files")

                nc = Dataset(
                    os.path.join(self.config.working_dir, "ex.nc"),
                    "w",
                    format="NETCDF4",
                )

                nc.createDimension("time", None)
                E = nc.createVariable("time", np.dtype("float32").char, ("time",))
                E.units = "yr"
                E.long_name = "time"
                E.axis = "T"
                E[0] = self.t.numpy()

                nc.createDimension("y", len(self.y))
                E = nc.createVariable("y", np.dtype("float32").char, ("y",))
                E.units = "m"
                E.long_name = "y"
                E.axis = "Y"
                E[:] = self.y.numpy()

                nc.createDimension("x", len(self.x))
                E = nc.createVariable("x", np.dtype("float32").char, ("x",))
                E.units = "m"
                E.long_name = "x"
                E.axis = "X"
                E[:] = self.x.numpy()

                for var in self.config.vars_to_save:

                    E = nc.createVariable(
                        var, np.dtype("float32").char, ("time", "y", "x")
                    )
                    E.long_name = self.var_info[var][0]
                    E.units = self.var_info[var][1]
                    E[0, :, :] = vars(self)[var].numpy()

                nc.close()

            else:

                nc = Dataset(
                    os.path.join(self.config.working_dir, "ex.nc"),
                    "a",
                    format="NETCDF4",
                )

                d = nc.variables["time"][:].shape[0]
                nc.variables["time"][d] = self.t.numpy()

                for var in self.config.vars_to_save:
                    nc.variables[var][d, :, :] = vars(self)[var].numpy()

                nc.close()

            self.tcomp["Outputs ncdf"][-1] -= time.time()
            self.tcomp["Outputs ncdf"][-1] *= -1
            
    def update_ncdf_ts(self, force=False):
        """
            Initialize  and write the ncdf time serie file
        """

        if not hasattr(self, "already_called_update_ncdf_ts"):
            self.already_called_update_ncdf_ts = True

        if force | self.saveresult:
             
            vol  = np.sum(self.thk)   * (self.dx ** 2) / 10 ** 9
            area = np.sum(self.thk>1) * (self.dx ** 2) / 10 ** 6

            if self.it == 0:

                if self.config.verbosity == 1:
                    print("Initialize NCDF output Files")

                nc = Dataset(
                    os.path.join(self.config.working_dir, "ts.nc"),
                    "w",
                    format="NETCDF4",
                )

                nc.createDimension("time", None)
                E = nc.createVariable("time", np.dtype("float32").char, ("time",))
                E.units = "yr"
                E.long_name = "time"
                E.axis = "T"
                E[0] = self.t.numpy()

                for var in ['vol','area']:
                    E = nc.createVariable( var, np.dtype("float32").char, ("time") )
                    E[0] = vars()[var].numpy()
                    E.long_name = self.var_info[var][0]
                    E.units = self.var_info[var][1]
                nc.close()

            else:

                nc = Dataset( os.path.join(self.config.working_dir, "ts.nc"), "a", format="NETCDF4" )
                d = nc.variables["time"][:].shape[0]

                nc.variables["time"][d] = self.t.numpy()
                for var in ['vol','area']:
                    nc.variables[var][d] = vars()[var].numpy()
                nc.close()

    ####################################################################################
    ####################################################################################
    ####################################################################################
    #                               COMPUTE T AND DT
    ####################################################################################
    ####################################################################################
    ####################################################################################

    def read_config_param_t_dt(self):

        # NUMERICL PARAMETER FOR TIME STEP
        self.parser.add_argument(
            "--cfl", type=float, default=0.3, help="CFL number must be below 1 (0.3)"
        )
        self.parser.add_argument(
            "--dtmax",
            type=float,
            default=10.0,
            help="Maximum time step, used only with slow ice (10.0)",
        )

    def update_t_dt(self):
        """
            compute time step to satisfy the CLF condition and hit requested saving times
        """
        if self.config.verbosity == 1:
            print("Update DT from the CFL condition at time : ", self.t.numpy())

        if not hasattr(self, "already_called_update_t_dt"):
            self.tcomp["Time step"] = []
            self.already_called_update_t_dt = True

            self.tsave = np.ndarray.tolist(
                np.arange(self.config.tstart, self.config.tend, self.config.tsave)
            ) + [self.config.tend]
            self.itsave = 0

        self.tcomp["Time step"].append(time.time())

        velomax = max(
            tf.math.reduce_max(tf.math.abs(self.ubar)),
            tf.math.reduce_max(tf.math.abs(self.vbar)),
        ).numpy()

        if velomax > 0:
            self.dt_target = min(self.config.cfl * self.dx / velomax, self.config.dtmax)
        else:
            self.dt_target = self.config.dtmax

        self.dt = self.dt_target

        if self.tsave[self.itsave + 1] <= self.t.numpy() + self.dt:
            self.dt = self.tsave[self.itsave + 1] - self.t.numpy()
            self.t.assign( self.tsave[self.itsave + 1] )
            self.saveresult = True
            self.itsave += 1
        else:
            self.t.assign( self.t.numpy() + self.dt )
            self.saveresult = False

        self.it += 1

        self.tcomp["Time step"][-1] -= time.time()
        self.tcomp["Time step"][-1] *= -1

    ####################################################################################
    ####################################################################################
    ####################################################################################
    #                               GET MAG AND GRADIENTS
    ####################################################################################
    ####################################################################################
    ####################################################################################

    @tf.function()
    def getmag(self, u, v):
        """
            return the norm of a 2D vector, e.g. to compute velbase_mag
        """
        return tf.norm(
            tf.concat([tf.expand_dims(u, axis=-1), tf.expand_dims(v, axis=-1)], axis=2),
            axis=2,
        )

    @tf.function()
    def compute_gradient_tf(self, s, dx, dy):
        """
            compute spatial 2D gradient of a given field
        """

        EX = tf.concat([s[:, 0:1], 0.5 * (s[:, :-1] + s[:, 1:]), s[:, -1:]], 1)
        diffx = (EX[:, 1:] - EX[:, :-1]) / dx

        EY = tf.concat([s[0:1, :], 0.5 * (s[:-1, :] + s[1:, :]), s[-1:, :]], 0)
        diffy = (EY[1:, :] - EY[:-1, :]) / dy

        return diffx, diffy

    ####################################################################################
    ####################################################################################
    ####################################################################################
    #                               ICEFLOW
    ####################################################################################
    ####################################################################################
    ####################################################################################

    def read_fields_and_bounds(self, path):
        """
            get fields (input and outputs) from given file
        """

        self.fieldbounds = {}
        self.fieldin = []
        self.fieldout = []

        fid = open(os.path.join(path, "fieldin.dat"), "r")
        for fileline in fid:
            part = fileline.split()
            self.fieldin.append(part[0])
            self.fieldbounds[part[0]] = float(part[1])
        fid.close()

        fid = open(os.path.join(path, "fieldout.dat"), "r")
        for fileline in fid:
            part = fileline.split()
            self.fieldout.append(part[0])
            self.fieldbounds[part[0]] = float(part[1])
        fid.close()

    def read_config_param_iceflow(self):

        self.parser.add_argument(
            "--model_lib_path",
            type=str,
            default="/home/jouvetg/IGM/model-lib/f12_cfsflow",
            help="model directory",
        )
        self.parser.add_argument(
            "--multiple_window_size",
            type=int,
            default=0,
            help="In case the mdel is a unet, it must force window size to be multiple of e.g. 8",
        )
        self.parser.add_argument(
            "--force_max_velbar",
            type=float,
            default=0,
            help="This permits to artificially upper-bound velocities, active if > 0",
        )

    def initialize_iceflow(self):
        """
            set-up the iceflow emulator
        """
 
        dirpath = os.path.join(self.config.model_lib_path, str(int(self.dx)))
     
        assert os.path.isdir(dirpath)
        
        self.read_fields_and_bounds(dirpath)

        self.iceflow_mapping = {}
        self.iceflow_mapping["fieldin"] = self.fieldin
        self.iceflow_mapping["fieldout"] = self.fieldout

        self.iceflow_model = tf.keras.models.load_model( os.path.join(dirpath, "model.h5") )

#        print(self.iceflow_model.summary())

        Ny = self.thk.shape[0]
        Nx = self.thk.shape[1]

        if self.config.multiple_window_size > 0:
            NNy = self.config.multiple_window_size * math.ceil(
                Ny / self.config.multiple_window_size
            )
            NNx = self.config.multiple_window_size * math.ceil(
                Nx / self.config.multiple_window_size
            )
            self.PAD = [[0, NNy - Ny], [0, NNx - Nx]]
        else:
            self.PAD = [[0, 0], [0, 0]]

        self.tcomp["Ice flow"] = []

    def update_iceflow(self):
        """
            function update the ice flow using the neural network emulator
        """

        if self.config.verbosity == 1:
            print("Update ICEFLOW at time : ", igm.t)

        self.tcomp["Ice flow"].append(time.time())

        X = tf.expand_dims(
            tf.stack(
                [
                    tf.pad(vars(self)[f], self.PAD, "CONSTANT") / self.fieldbounds[f]
                    for f in self.iceflow_mapping["fieldin"]
                ],
                axis=-1,
            ),
            axis=0,
        )

        Y = self.iceflow_model.predict_on_batch(X)

        Ny, Nx = self.thk.shape
        for kk, f in enumerate(self.iceflow_mapping["fieldout"]):
            vars(self)[f].assign(
                tf.where(self.thk > 0, Y[0, :Ny, :Nx, kk], 0) * self.fieldbounds[f]
            )
         
        if (self.config.force_max_velbar>0):
        
            self.velbar_mag = self.getmag(self.ubar, self.vbar)
            
            self.ubar.assign( 
              tf.where( self.velbar_mag >= self.config.force_max_velbar,
                        self.config.force_max_velbar * ( self.ubar / self.velbar_mag ),
                        self.ubar
                      )
                            )
            self.vbar.assign( 
              tf.where( self.velbar_mag >= self.config.force_max_velbar,
                        self.config.force_max_velbar * ( self.vbar / self.velbar_mag ),
                        self.vbar
                      )
                            )
                     
        self.tcomp["Ice flow"][-1] -= time.time()
        self.tcomp["Ice flow"][-1] *= -1

    ####################################################################################
    ####################################################################################
    ####################################################################################
    #                              CLIMATE
    ####################################################################################
    ####################################################################################
    ####################################################################################

    def read_config_param_climate(self):

        # CLIMATE PARAMETERS
        self.parser.add_argument(
            "--clim_update_freq",
            type=float,
            default=1,
            help="Update the climate each X years (1)",
        )
        self.parser.add_argument(
            "--type_climate", type=str, default="", help="toy or any custom climate",
        )

    def update_climate(self, force=False):
        """
            compute climate at time t
        """

        if len(self.config.type_climate) > 0:

            if not hasattr(self, "already_called_update_climate"):

                getattr(self, "load_climate_data_" + self.config.type_climate)()
                self.tlast_clim = -1.0e5000
                self.tcomp["Climate"] = []
                self.already_called_update_climate = True

            new_clim_needed = (self.t.numpy() - self.tlast_clim) >= self.config.clim_update_freq

            if force | new_clim_needed:

                if self.config.verbosity == 1:
                    print("Construct climate at time : ", igm.t)

                self.tcomp["Climate"].append(time.time())

                getattr(self, "update_climate_" + self.config.type_climate)()

                self.tlast_clim = self.t.numpy()

                self.tcomp["Climate"][-1] -= time.time()
                self.tcomp["Climate"][-1] *= -1

    ####################################################################################
    ####################################################################################
    ####################################################################################
    #                                 MASS BALANCE
    ####################################################################################
    ####################################################################################
    ####################################################################################

    def read_config_param_smb(self):

        self.parser.add_argument(
            "--mb_update_freq",
            type=float,
            default=1,
            help="Update the mass balance each X years (1)",
        )
        self.parser.add_argument(
            "--type_mass_balance",
            type=str,
            default="simple",
            help="zero, simple, given",
        )
        self.parser.add_argument(
            "--mb_scaling", type=float, default=1.0, help="mass balance scaling"
        )
        self.parser.add_argument(
            "--mb_simple_file",
            type=str,
            default="mb_simple_param.txt",
            help="mb_simple_file",
        )

    def init_smb_simple(self):
        """
            initialize simple mass balance
        """
        param = np.loadtxt(
            os.path.join(self.config.working_dir, self.config.mb_simple_file),
            skiprows=1,
            dtype=np.float32,
        )

        self.gradabl = interp1d(
            param[:, 0],
            param[:, 1],
            fill_value=(param[0, 1], param[-1, 1]),
            bounds_error=False,
        )
        self.gradacc = interp1d(
            param[:, 0],
            param[:, 2],
            fill_value=(param[0, 2], param[-1, 2]),
            bounds_error=False,
        )
        self.ela = interp1d(
            param[:, 0],
            param[:, 3],
            fill_value=(param[0, 3], param[-1, 3]),
            bounds_error=False,
        )
        self.maxacc = interp1d(
            param[:, 0],
            param[:, 4],
            fill_value=(param[0, 4], param[-1, 4]),
            bounds_error=False,
        )

    def update_smb_simple(self):
        """
            mass balance 'simple' parametrized by ELA, ablation and accumulation gradients, and max acuumulation
        """

        ela = np.float32(self.ela(self.t))
        gradabl = np.float32(self.gradabl(self.t))
        gradacc = np.float32(self.gradacc(self.t))
        maxacc = np.float32(self.maxacc(self.t))

        smb = self.usurf - ela
        smb *= tf.where(tf.less(smb, 0), gradabl, gradacc)
        smb = tf.clip_by_value(smb, -100, maxacc)
        smb = tf.where(self.icemask > 0.5, smb, -10)

        self.smb.assign(smb)

    def update_smb(self, force=False):
        """
            update_mass balance
        """

        if not hasattr(self, "already_called_update_smb"):
            self.tlast_mb = -1.0e5000
            self.tcomp["Mass balance"] = []
            if len(self.config.type_mass_balance) > 0:
                if hasattr(self, "init_smb_" + self.config.type_mass_balance):
                     getattr(self, "init_smb_" + self.config.type_mass_balance)()
            self.already_called_update_smb = True

        if (force) | ((self.t.numpy() - self.tlast_mb) >= self.config.mb_update_freq):

            if self.config.verbosity == 1:
                print("Construct mass balance at time : ", self.t.numpy())

            self.tcomp["Mass balance"].append(time.time())

            if len(self.config.type_mass_balance) > 0:
                getattr(self, "update_smb_" + self.config.type_mass_balance)()
            else:
                self.smb.assign(tf.zeros_like(self.topg))

            if hasattr(self, "icemask"):
                self.smb.assign(self.smb * self.icemask)

            if not self.config.mb_scaling == 1:
                self.smb.assign(self.smb * self.config.mb_scaling)

            self.tlast_mb = self.t.numpy()

            if self.config.stop:
                mb_np = self.smb.numpy()

            self.tcomp["Mass balance"][-1] -= time.time()
            self.tcomp["Mass balance"][-1] *= -1

    ####################################################################################
    ####################################################################################
    ####################################################################################
    #                              TRANSPORT
    ####################################################################################
    ####################################################################################
    ####################################################################################

    def update_thk(self):
        """
            update ice thickness solving dh/dt + d(u h)/dx + d(v h)/dy = f using 
            upwind finite volume, update usurf and slopes
        """

        if not hasattr(self, "already_called_update_icethickness"):
            self.tcomp["Transport"] = []
            self.already_called_update_icethickness = True

        if self.config.verbosity == 1:
            print("Ice thickness equation at time : ", self.t.numpy())

        self.tcomp["Transport"].append(time.time())

        # compute the divergence of the flux
        self.divflux = self.compute_divflux(
            self.ubar, self.vbar, self.thk, self.dx, self.dx
        )

        # Forward Euler with projection to keep ice thickness non-negative
        self.thk.assign(tf.maximum(self.thk + self.dt * (self.smb - self.divflux), 0))

        self.usurf.assign(self.topg + self.thk)

        self.slopsurfx, self.slopsurfy = self.compute_gradient_tf(
            self.usurf, self.dx, self.dx
        )

        self.tcomp["Transport"][-1] -= time.time()
        self.tcomp["Transport"][-1] *= -1

    @tf.function()
    def compute_divflux(self, u, v, h, dx, dy):
        """
        #   upwind computation of the divergence of the flux : d(u h)/dx + d(v h)/dy
        #   First, u and v are computed on the staggered grid (i.e. cell edges)
        #   Second, one extend h horizontally by a cell layer on any bords (assuming same value)
        #   Third, one compute the flux on the staggered grid slecting upwind quantities
        #   Last, computing the divergence on the staggered grid yields values def on the original grid
        """
        
        ## Compute u and v on the staggered grid
        u = tf.concat([u[:, 0:1], 0.5 * (u[:, :-1] + u[:, 1:]), u[:, -1:]], 1) # has shape (ny,nx+1)
        v = tf.concat([v[0:1, :], 0.5 * (v[:-1, :] + v[1:, :]), v[-1:, :]], 0) # has shape (ny+1,nx)

        # Extend h with constant value at the domain boundaries
        Hx = tf.pad(h, [[0, 0], [1, 1]], "CONSTANT")    # has shape (ny,nx+2)
        Hy = tf.pad(h, [[1, 1], [0, 0]], "CONSTANT")    # has shape (ny+2,nx)

        ## Compute fluxes by selcting the upwind quantities
        Qx = u * tf.where(u > 0, Hx[:, :-1], Hx[:, 1:]) # has shape (ny,nx+1)
        Qy = v * tf.where(v > 0, Hy[:-1, :], Hy[1:, :]) # has shape (ny+1,nx)

        ## Computation of the divergence, final shape is (ny,nx)
        return (Qx[:, 1:] - Qx[:, :-1]) / dx + (Qy[1:, :] - Qy[:-1, :]) / dy

    ####################################################################################
    ####################################################################################
    ####################################################################################
    #                              PLOT
    ####################################################################################
    ####################################################################################
    ####################################################################################

    def read_config_param_plot(self):

        self.parser.add_argument(
            "--varplot", type=str, default="velbar_mag", help="variable to plot",
        )
        self.parser.add_argument(
            "--varplot_max",
            type=float,
            default=500,
            help="maximum value of the varplot variable used to adjust the scaling of the colorbar",
        )

    def update_plot(self, force=False):
        """
            Plot thickness, velocity, mass balance
        """

        if force | (self.saveresult & self.config.plot_result):

            firstime = False
            if not hasattr(self, "already_called_update_plot"):
                self.already_called_update_plot = True
                self.tcomp["Outputs plot"] = []
                firstime = True

            self.tcomp["Outputs plot"].append(time.time())

            if self.config.varplot == "velbar_mag":
                self.velbar_mag = self.getmag(self.ubar, self.vbar)

            if firstime:

                self.fig = plt.figure(dpi=200)
                self.ax = self.fig.add_subplot(1, 1, 1)
                self.ax.axis("off")
                im = self.ax.imshow(
                    vars(self)[self.config.varplot],
                    origin="lower",
                    cmap="viridis",
                    vmin=0,
                    vmax=self.config.varplot_max,
                )
                self.ax.set_title("YEAR : " + str(self.t.numpy()), size=15)
                self.cbar = plt.colorbar(im)

            else:
                im = self.ax.imshow(
                    vars(self)[self.config.varplot],
                    origin="lower",
                    cmap="viridis",
                    vmin=0,
                    vmax=self.config.varplot_max,
                )
                self.ax.set_title("YEAR : " + str(self.t.numpy()), size=15)

            if self.config.plot_live:
                clear_output(wait=True)
                display(self.fig)

            else:
                plt.savefig(
                    os.path.join(
                        self.config.working_dir,
                        self.config.varplot + "-" + str(self.t.numpy()).zfill(4) + ".png",
                    ),
                    bbox_inches="tight",
                    pad_inches=0.2,
                )

            self.tcomp["Outputs plot"][-1] -= time.time()
            self.tcomp["Outputs plot"][-1] *= -1

    def plot_computational_pie(self):
        """
            Plot to the computational time of each model components in a pie
        """

        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                return "{:.0f}".format(val)

            return my_autopct

        total = []
        name = []

        for i, key in enumerate(self.tcomp.keys()):
            if not key == "All":
                total.append(np.sum(self.tcomp[key][1:]))
                name.append(key)

        sumallindiv = np.sum(total)

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"), dpi=200)
        wedges, texts, autotexts = ax.pie(
            total, autopct=make_autopct(total), textprops=dict(color="w")
        )
        ax.legend(
            wedges,
            name,
            title="Model components",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
        )
        plt.setp(autotexts, size=8, weight="bold")
        #    ax.set_title("Matplotlib bakery: A pie")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.working_dir, "PIE-COMPUTATIONAL.png"), pad_inches=0
        )
        plt.close("all")

    ####################################################################################
    ####################################################################################
    ####################################################################################
    #                              PRINT INFO
    ####################################################################################
    ####################################################################################
    ####################################################################################

    def print_info(self):
        """
            This serves to print key info on the fly during computation
        """
        if self.saveresult:
            print(
                "IGM %s : Iterations = %6.0f  |  Time = %8.0f  |  DT = %7.2f  |  Ice Volume (km^3) = %10.2f "
                % (
                    datetime.datetime.now().strftime("%H:%M:%S"),
                    self.it,
                    self.t,
                    self.dt_target,
                    np.sum(self.thk) * (self.dx ** 2) / 10 ** 9,
                )
            )

    def print_comp_info_live(self):
        """
            This serves to print computational info on the fly during computation
        """
        if self.saveresult:
            for key in self.tcomp.keys():
                CELA = (
                    key,
                    np.mean(self.tcomp[key][2:]),
                    np.sum(self.tcomp[key][2:]),
                    len(self.tcomp[key][2:]),
                )
                print(
                    "     %15s  |  mean time per it : %8.4f  |  total : %8.4f  |  number it  : %8.0f"
                    % CELA
                )

    def print_all_comp_info(self):
        """
            This serves to print computational info report
        """

        print("Computational statistics report:")
        with open(
            os.path.join(self.config.working_dir, "computational-statistics.txt"), "w"
        ) as f:
            for key in self.tcomp.keys():
                CELA = (
                    key,
                    np.mean(self.tcomp[key][2:]),
                    np.sum(self.tcomp[key][2:]),
                    len(self.tcomp[key][2:]),
                )
                print(
                    "     %15s  |  mean time per it : %8.4f  |  total : %8.4f  |  number it : %8.0f"
                    % CELA,
                    file=f,
                )
                print(
                    "     %15s  |  mean time per it : %8.4f  |  total : %8.4f  |  number it  : %8.0f"
                    % CELA
                )

    ####################################################################################
    ####################################################################################
    ####################################################################################
    #                              RUN
    ####################################################################################
    ####################################################################################
    ####################################################################################

    def run(self):

        self.initialize()

        with tf.device(self.device_name):

            self.load_ncdf_data(self.config.geology_file)

            self.initialize_fields()

            if len(self.config.restartingfile) > 0:
                self.restart(-1)

            self.initialize_iceflow()

            self.update_climate()

            self.update_smb()

            if self.config.optimize:
                self.optimize()

            self.update_iceflow()

            self.update_ncdf_ex()
            
            self.update_ncdf_ts()

            self.print_info()

            while self.t.numpy() < self.config.tend:

                self.tcomp["All"].append(time.time())

                self.update_climate()

                self.update_smb()

                self.update_iceflow()

                self.update_t_dt()

                self.update_thk()

                if self.config.update_topg:
                    self.update_topg()

                self.update_ncdf_ex()
                
                self.update_ncdf_ts()

                self.update_plot()

                self.print_info()

                self.tcomp["All"][-1] -= time.time()
                self.tcomp["All"][-1] *= -1

        self.print_all_comp_info()

####################################################################################
####################################################################################
####################################################################################
#   END   END   END   END   END   END   END   END   END   END   END   END   END
####################################################################################
####################################################################################
####################################################################################
