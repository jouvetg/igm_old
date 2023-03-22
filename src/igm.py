#!/usr/bin/env python3

"""
Copyright (C) 2021-2022 Guillaume Jouvet <guillaume.jouvet@geo.uzh.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

# This file contains all core functions the Instructed Glacier Model (IGM),
# functions that are sorted thematic-wise as follows:

#      INITIALIZATION : contains all to initialize variables
#      I/O NCDF : Files for data Input/Output using NetCDF file format
#      COMPUTE T AND DT : Function to compute adaptive time-step and time
#      ICEFLOW : Containt the function that serve compute the emulated iceflow
#      CLIMATE : Templates for function updating climate
#      MASS BALANCE : Simple mass balance
#      EROSION : Erosion law
#      TRANSPORT : This handles the mass conservation
#      TRACKING ; this permits to computa indivudal trajecotties
#      3DVEL  : This permits to export reconsctruted 3D ice velocity field
#      OPTI : All for the optimization / data assimilation
#      PLOT : Plotting functions
#      PRINT INFO : handle the printing of output during computation
#      RUN : the main function that wrap all functions within an time-iterative loop

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
from scipy import stats

def str2bool(v):
    return v.lower() in ("true", "1")

# @tf.keras.utils.register_keras_serializable()
# class StackGlen(tf.keras.layers.Layer):
    
#     def __init__(self,nb_outputs,**kwargs):
#         super(StackGlen, self).__init__()
#         self.nb_outputs = nb_outputs
        
#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({"nb_outputs": self.nb_outputs})
#         return config
 
#     def call(self, inputs):
          
#         U = inputs[:,:,:,:int(self.nb_outputs/2)]
#         V = inputs[:,:,:,int(self.nb_outputs/2):]
#         uvelbase = U[:, :, :, 0]
#         vvelbase = V[:, :, :, 0]
#         ubar     = tf.reduce_mean(U, axis=-1) 
#         vbar     = tf.reduce_mean(V, axis=-1)
#         uvelsurf = U[:, :, :, -1]
#         vvelsurf = V[:, :, :, -1]

#         return tf.stack([uvelbase,vvelbase,ubar,vbar,uvelsurf,vvelsurf],axis=-1)

####################################################################################

class Igm:

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
        
        # Read parameters
        self.parser = argparse.ArgumentParser(description="IGM")
        self.read_config_param()
        self.config = self.parser.parse_args()

        # These 3 lines were found necessary to ensure compatibility with newest TF
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)
                
        print('Build IGM class')

    def read_config_param(self):
        """
        LIst of the main IGM parameters (some others are defined in other functions)
        """

        self.parser.add_argument(
            "--working_dir", 
            type=str, default="", 
            help="Working directory (default empty string)"
        )
        self.parser.add_argument(
            "--geology_file", 
            type=str, 
            default="geology.nc", 
            help="Geology input file (default: geology.nc)"
        )
        self.parser.add_argument(
            "--resample", 
            type=int, 
            default=1, 
            help="Resample the data of ncdf data file to a coarser resolution (default: 1)"
        )
        self.parser.add_argument(
            "--tstart", 
            type=float, 
            default=0.0, 
            help="Start modelling time (default 0)"
        )
        self.parser.add_argument(
            "--tend", 
            type=float, 
            default=100.0, 
            help="End modelling time (default: 100)"
        )
        self.parser.add_argument(
            "--restartingfile",
            type=str,
            default="",
            help="Provide restarting file if no empty string (default: empty string)",
        )
        self.parser.add_argument(
            "--verbosity",
            type=int,
            default=0,
            help="Verbosity level of IGM (default: 0 = no verbosity)",
        )
        self.parser.add_argument(
            "--tsave", 
            type=float, 
            default=10, 
            help="Save result each X years (default: 10)"
        )
        self.parser.add_argument(
            "--plot_result", 
            type=str2bool, 
            default=False, 
            help="Save plot results in png (default: False)",
        )
        self.parser.add_argument(
            "--plot_live",
            type=str2bool,
            default=False,
            help="Display plots live the results during computation (Default: False)",
        )
        self.parser.add_argument(
            "--usegpu",
            type=str2bool,
            default=True,
            help="Use the GPU instead of CPU (default: True)",
        )
        self.parser.add_argument(
            "--stop",
            type=str2bool,
            default=False,
            help="experimental, just to get fair comp time, to be removed ....",
        )
        self.parser.add_argument(
            "--init_strflowctrl", 
            type=float, 
            default=78, 
            help="Initial strflowctrl (default 78)",
        )
        self.parser.add_argument(
            "--init_slidingco", 
            type=float, 
            default=0, 
            help="Initial sliding coeeficient slidingco (default: 0)",
        )
        self.parser.add_argument(
            "--init_arrhenius", 
            type=float, 
            default=78, 
            help="Initial arrhenius factor arrhenuis (default: 78)",
        )
        
        # Read all parameters defined in function read_config_param_*
        for p in dir(self):
            if "read_config_param_" in p:
                getattr(self, p)()

    ##################################################

    def initialize(self):
        """
        function initialize the strict minimum for IGM class, and print parameters
        """

        print(
            "+++++++++++++++++++ START IGM ++++++++++++++++++++++++++++++++++++++++++"
        )

        # Initialize time, time step, saving result flag, ...
        self.t = tf.Variable(float(self.config.tstart))
        self.it = 0
        self.dt = float(self.config.dtmax)
        self.dt_target = float(self.config.dtmax)

        self.saveresult = True

        # All statisitics of the computational times taken by any components will be stored here
        self.tcomp = {}
        self.tcomp["All"] = []

        # Define the device name to perform computations (CPU or GPU)
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

        # Print parameters in screen and a dedicated file
        with open(
            os.path.join(self.config.working_dir, "igm-run-parameters.txt"), "w"
        ) as f:
            print("PARAMETERS ARE ...... ")
            for ck in self.config.__dict__:
                print("%30s : %s" % (ck, self.config.__dict__[ck]))
                print("%30s : %s" % (ck, self.config.__dict__[ck]), file=f)

        # prepare the clean.sh file to remove all files created in one run
        os.system('echo rm -r '+ os.path.join(self.config.working_dir, "__pycache__") + ' > clean.sh')
        os.system('echo rm '+ os.path.join(self.config.working_dir, "igm-run-parameters.txt") + ' >> clean.sh')

    def load_ncdf_data(self, filename):
        """
        Load the input files from netcdf file
        """
        if self.config.verbosity == 1:
            print("LOAD NCDF file")

        nc = Dataset(os.path.join(self.config.working_dir, filename), "r")

        x = np.squeeze(nc.variables["x"]).astype("float32")
        y = np.squeeze(nc.variables["y"]).astype("float32")
        
        # make sure the grid has same cell spacing in x and y
        assert x[1] - x[0] == y[1] - y[0]

        # load any field contained in the ncdf file, replace missing entries by nan
        for var in nc.variables:
            if not var in ["x", "y"]:
                vars()[var] = np.squeeze(nc.variables[var]).astype("float32")
                vars()[var] = np.where(vars()[var] > 10 ** 35, np.nan, vars()[var])

        # resample if requested
        if self.config.resample > 1:
            xx = x[:: self.config.resample]
            yy = y[:: self.config.resample]
            for var in nc.variables:
                if not var in ["x", "y"]:
                    vars()[var] = RectBivariateSpline(y, x, vars()[var])(yy, xx)
            x = xx
            y = yy

        # transform from numpy to tensorflow
        for var in nc.variables:
            if var in ["x", "y"]:
                vars(self)[var] = tf.constant(vars()[var].astype("float32"))
            else:
                vars(self)[var] = tf.Variable(vars()[var].astype("float32"))

        nc.close()

    def initialize_fields(self):
        """
        Initialize fields, complete the loading of geology
        """

        if self.config.verbosity == 1:
            print("Initialize fields")

        # at this point, we should have defined at least x, y
        assert hasattr(self, "x")
        assert hasattr(self, "y")

        if hasattr(self, "usurfobs"):
            self.usurf = tf.Variable(self.usurfobs)

        # if thickness is not defined in the netcdf, then it is set to zero
        if not hasattr(self, "thk"):
            self.thk = tf.Variable(tf.zeros((self.y.shape[0], self.x.shape[0])))

        # at this point, we should have defined at least topg or usurf
        assert hasattr(self, "topg") | hasattr(self, "usurf")

        # define usurf (or topg) from topg (or usurf) and thk
        if hasattr(self, "usurf"):
            self.topg  = tf.Variable(self.usurf - self.thk)
        else:
            self.usurf = tf.Variable(self.topg + self.thk)
            
        # if no icemask, create a mask with one everywhere
        if not hasattr(self, "icemask"):
            self.icemask = tf.Variable(tf.ones_like(self.thk))

        # Here we initialize to zero a set of variables used later on
        var_to_init  = ["uvelsurf","vvelsurf","wvelsurf"]
        var_to_init += ["uvelbase","vvelbase","wvelbase","ubar","vbar"]
        var_to_init += ["smb","dhdt","divflux"] # TODO: remove dhdt?
        
        for var in var_to_init:
            if not hasattr(self, var):
                vars(self)[var] = tf.Variable(tf.zeros_like(self.thk))

        # here we initialize variable parmaetrizing ice flow
        if not hasattr(self, "strflowctrl"):
            self.strflowctrl = tf.Variable(
                tf.ones_like(self.thk) * self.config.init_strflowctrl
            )

        if not hasattr(self, "arrhenius"):
            self.arrhenius = tf.Variable(
                tf.ones_like(self.thk) * self.config.init_arrhenius
            )

        if not hasattr(self, "slidingco"):
            self.slidingco = tf.Variable(
                tf.ones_like(self.thk) * self.config.init_slidingco
            )
 
        # define grids, i.e. self.X and self.Y has same shape as self.thk
        self.X, self.Y = tf.meshgrid(self.x, self.y)

        # define cell spacing
        self.dx = self.x[1] - self.x[0]

        # define dX
        self.dX = tf.ones_like(self.thk) * self.dx

        # compiute surface gradient
        self.slopsurfx, self.slopsurfy = self.compute_gradient_tf(
            self.usurf, self.dx, self.dx
        )

        # give information on variables for output ncdf, TODO: IMPROVE
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
        self.var_info["wvelsurf"] = ["z surface velocity of ice", "m/y"]
        self.var_info["velsurf_mag"] = ["Surface velocity magnitude of ice", "m/y"]
        self.var_info["uvelbase"] = ["x basal velocity of ice", "m/y"]
        self.var_info["vvelbase"] = ["y basal velocity of ice", "m/y"]
        self.var_info["wvelbase"] = ["z basal velocity of ice", "m/y"]
        self.var_info["velbase_mag"] = ["Basal velocity magnitude of ice", "m/y"]
        self.var_info["divflux"] = ["Divergence of the ice flux", "m/y"]
        self.var_info["strflowctrl"] = ["arrhenius+1.0*slidingco","MPa$^{-3}$ a$^{-1}$"]
        self.var_info["dtopgdt"] = ["Erosion rate", "m/y"]
        self.var_info["arrhenius"] = ["Arrhenius factor", "MPa$^{-3}$ a$^{-1}$"]
        self.var_info["slidingco"] = ["Sliding Coefficient", "km MPa$^{-3}$ a$^{-1}$"]
        self.var_info["meantemp"] = ["Mean anual surface temperatures", "°C"]
        self.var_info["meanprec"] = ["Mean anual precipitation", "m/y"]
        self.var_info["vol"] = ["Ice volume", "km^3"]
        self.var_info["area"] = ["Glaciated area", "km^2"]
        self.var_info["velsurfobs_mag"] = ["Obs. surf. speed of ice","m/y"]
        self.var_info["weight_particles"] = ["weight_particles","no"]

    def restart(self):
        """
        Permit to restart a simulation from a ncdf file y taking the last iterate
        """
        if self.config.verbosity == 1:
            print("READ RESTARTING FILE, OVERIDE FIELDS WHEN GIVEN")

        nc = Dataset(self.config.restartingfile, "r")

        Rx = tf.constant(np.squeeze(nc.variables["x"]).astype("float32"))
        Ry = tf.constant(np.squeeze(nc.variables["y"]).astype("float32"))

        # check consistency between original ncdf and restart file
        assert Rx.shape == self.x.shape
        assert Ry.shape == self.y.shape

        # load all variables
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
        """
        this is a simple function that serves to monitor the space in Mbyte taken by a variable
        """
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
            default=["topg", "usurf", "thk", "smb", "velbar_mag", "velsurf_mag",],
            help="List of variables to be recorded in the ncdf file",
        )

    def update_ncdf_ex(self):
        """
        This function write 2D field variables defined in the list config.vars_to_save into the ncdf output file ex.nc
        """

        if not hasattr(self, "already_called_update_ncdf_ex"):
            self.tcomp["Outputs ncdf"] = []
            self.already_called_update_ncdf_ex = True

        if self.saveresult:

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
                
                os.system('echo rm '+ os.path.join(self.config.working_dir, "ex.nc") + ' >> clean.sh')

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

    def update_ncdf_ts(self):
        """
        This function write time serie variables (ice glaziated area and volume) into the ncdf output file ts.nc
        """

        if not hasattr(self, "already_called_update_ncdf_ts"):
            self.already_called_update_ncdf_ts = True

        if self.saveresult:

            vol = np.sum(self.thk) * (self.dx ** 2) / 10 ** 9
            area = np.sum(self.thk > 1) * (self.dx ** 2) / 10 ** 6

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

                for var in ["vol", "area"]:
                    E = nc.createVariable(var, np.dtype("float32").char, ("time"))
                    E[0] = vars()[var].numpy()
                    E.long_name = self.var_info[var][0]
                    E.units = self.var_info[var][1]
                nc.close()
                
                os.system('echo rm '+ os.path.join(self.config.working_dir, "ts.nc") + ' >> clean.sh')

            else:

                nc = Dataset(
                    os.path.join(self.config.working_dir, "ts.nc"),
                    "a",
                    format="NETCDF4",
                )
                d = nc.variables["time"][:].shape[0]

                nc.variables["time"][d] = self.t.numpy()
                for var in ["vol", "area"]:
                    nc.variables[var][d] = vars()[var].numpy()
                nc.close()
                
    def load_tif_data(self,variables):
        """
        Load the input files from tiff file (alternative to load_ncdf_data)
        Select available fields in variables
        you need at least topg or usurf, and thk,
        filed e.g. topg.tif, thk.tif must be present in the working forlder
        you need rasterio to play this function
        """
        
        import rasterio 
        
        for var in variables:
            file = os.path.join(self.config.working_dir,var+'.tif')
            if os.path.exists(file):
                self.profile_tif_file = rasterio.open(file, 'r').profile
                with rasterio.open(file) as src:    
                    vars()[var] = np.flipud(src.read(1))
                    height      = vars()[var].shape[0]
                    width       = vars()[var].shape[1]
                    cols, rows  = np.meshgrid(np.arange(width), np.arange(height))
                    x, y = rasterio.transform.xy(src.transform, rows, cols)
                    x = np.array(x)[0,:]
                    y = np.flip(np.array(y)[:,0])                  
                    vars(self)[var] = tf.Variable(vars()[var].astype("float32"))
                del src
 
        self.x = tf.constant(x.astype("float32"))
        self.y = tf.constant(y.astype("float32"))
 
    def update_tif_ex(self,variables):
        """
        Save variables in tiff file (alternative to update_ncdf_ex)
        Select available fields in variables, e.g. topg, usurf, ...
        Files will be created with names like thk-000040.tif in the workin direc.
        you need rasterio to play this function
        """
        
        import rasterio
        
        if self.saveresult:

            for var in variables:

                file = os.path.join(self.config.working_dir,var+'-'+str(int(self.t)).zfill(6)+'.tif')

                with rasterio.open(file,mode="w", **self.profile_tif_file) as src:
                     src.write(np.flipud(vars(self)[var]), 1)

                del src

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
            "--cfl", 
            type=float, 
            default=0.3, 
            help="CFL number for the stability of the mass conservation scheme, it must be below 1 (Default: 0.3)"
        )
        self.parser.add_argument(
            "--dtmax",
            type=float,
            default=10.0,
            help="Maximum time step allowed, used only with slow ice (default: 10.0)",
        )

    def update_t_dt(self):
        """
        Function update_t_dt() return time step dt (computed to satisfy the CFL condition), updated time t, and a boolean telling whether results must be saved or not. For stability reasons of the transport scheme for the ice thickness evolution, the time step must respect a CFL condition, controlled by parameter glacier.config.cfl, which is the maximum number of cells crossed in one iteration (this parameter cannot exceed one). 
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

        else:
            self.tcomp["Time step"].append(time.time())

            # compute maximum ice velocitiy magnitude
            velomax = max(
                tf.math.reduce_max(tf.math.abs(self.ubar)),
                tf.math.reduce_max(tf.math.abs(self.vbar)),
            ).numpy()

            # dt_target account for both cfl and dt_max
            if velomax > 0:
                self.dt_target = min(
                    self.config.cfl * self.dx / velomax, self.config.dtmax
                )
            else:
                self.dt_target = self.config.dtmax

            self.dt = self.dt_target
            
            # modify dt such that times of requested savings are reached exactly
            if self.tsave[self.itsave + 1] <= self.t.numpy() + self.dt:
                self.dt = self.tsave[self.itsave + 1] - self.t.numpy()
                self.t.assign(self.tsave[self.itsave + 1])
                self.saveresult = True
                self.itsave += 1
            else:
                self.t.assign(self.t.numpy() + self.dt)
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

        # EX = tf.concat([s[:, 0:1], 0.5 * (s[:, :-1] + s[:, 1:]), s[:, -1:]], 1)
        # diffx = (EX[:, 1:] - EX[:, :-1]) / dx

        # EY = tf.concat([s[0:1, :], 0.5 * (s[:-1, :] + s[1:, :]), s[-1:, :]], 0)
        # diffy = (EY[1:, :] - EY[:-1, :]) / dy
        
        EX = tf.concat([ 1.5*s[:,0:1] - 0.5*s[:,1:2], 0.5*s[:,:-1] + 0.5*s[:,1:], 1.5*s[:,-1:] - 0.5*s[:,-2:-1] ], 1)
        diffx = (EX[:, 1:] - EX[:, :-1]) / dx

        EY = tf.concat([ 1.5*s[0:1,:] - 0.5*s[1:2,:], 0.5*s[:-1,:] + 0.5*s[1:,:], 1.5*s[-1:,:] - 0.5*s[-2:-1,:] ], 0)
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

        fieldbounds = {}
        fieldin = []
        fieldout = []

        fid = open(os.path.join(path, "fieldin.dat"), "r")
        for fileline in fid:
            part = fileline.split()
            fieldin.append(part[0])
            fieldbounds[part[0]] = float(part[1])
        fid.close()

        fid = open(os.path.join(path, "fieldout.dat"), "r")
        for fileline in fid:
            part = fileline.split()
            fieldout.append(part[0])
            fieldbounds[part[0]] = float(part[1])
        fid.close()

        return fieldin, fieldout, fieldbounds

    def read_config_param_iceflow(self):

        self.parser.add_argument(
            "--iceflow_model_lib_path",
            type=str,
            default="../../model-lib/f15_cfsflow_GJ_22_a",
            help="Directory path of the deep-learning ice flow model",
        )
        self.parser.add_argument(
            "--multiple_window_size",
            type=int,
            default=0,
            help="In case the ANN is a U-net, it must force window size to be multiple of 2**N (default: 0)",
        )
        self.parser.add_argument(
            "--force_max_velbar",
            type=float,
            default=0,
            help="This permits to artificially upper-bound velocities, active if > 0 (default: 0)",
        )

    def initialize_iceflow(self):
        """
        set-up the iceflow emulator
        """

        dirpath = os.path.join(self.config.iceflow_model_lib_path, str(int(self.dx)))

        if not os.path.isdir(dirpath):
            dirpath = self.config.iceflow_model_lib_path

        # fieldin, fieldout, fieldbounds contains name of I/O variables, and bounds for scaling
        fieldin, fieldout, fieldbounds = self.read_fields_and_bounds(dirpath)

        self.iceflow_mapping = {}
        self.iceflow_mapping["fieldin"] = fieldin
        self.iceflow_mapping["fieldout"] = fieldout
        self.iceflow_fieldbounds = fieldbounds

        self.iceflow_model = tf.keras.models.load_model(
            os.path.join(dirpath, "model.h5")
        )

        # print(self.iceflow_model.summary())

        Ny = self.thk.shape[0]
        Nx = self.thk.shape[1]
        
        # In case of a U-net, must make sure the I/O size is multiple of 2**N
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

    def update_iceflow(self):
        """
        Ice flow dynamics are modeled using Artificial Neural Networks trained from physical models.
        
        You may find trained and ready-to-use ice flow emulators in the folder `model-lib/T_M_I_Y_V/R/`, where 'T_M_I_Y_V' defines the emulator, and R defines the spatial resolution. Make sure that the resolution of the picked emulator is available in the database. Results produced with IGM will strongly rely on the chosen emulator. Make sure that you use the emulator within the hull of its training dataset (e.g., do not model an ice sheet with an emulator trained with mountain glaciers) to ensure reliability (or fidelity w.r.t to the instructor model) -- the emulator is probably much better at interpolating than at extrapolating. Information on the training dataset is provided in a dedicated README coming along with the emulator.
        
        At the time of writing, I recommend using *f15_cfsflow_GJ_22_a*, which takes ice thickness, top surface slopes, the sliding coefficient c ('slidingco'), and Arrhenuis factor A ('arrhenius'), and return basal, vertical-average and surface x- and y- velocity components as depicted on the following graph:
        
        ![](https://github.com/jouvetg/igm/blob/main/fig/mapping-f15.png)
        
        I have trained *f15_cfsflow_GJ_22_a* using a large dataset of modeled glaciers (based on a Stokes-based CfsFlow ice flow solver) and varying sliding coefficient c, and Arrhenius factor A into a 2D space. 
        
        For now, only the emulator trained by CfsFlow and PISM is available with different resolutions. Consider training your own with the [Deep Learning Emulator](https://github.com/jouvetg/dle) if none of these emulators fill your need.
        """

        if self.config.verbosity == 1:
            print("Update ICEFLOW at time : ", self.t)

        if not hasattr(self, "already_called_update_iceflow"):

            self.tcomp["Ice flow"] = []
            self.already_called_update_iceflow = True
            self.initialize_iceflow()

        self.tcomp["Ice flow"].append(time.time())

        # Define the input of the NN, include scaling
        X = tf.expand_dims(
            tf.stack(
                [
                    tf.pad(vars(self)[f], self.PAD, "CONSTANT")
                    / self.iceflow_fieldbounds[f]
                    for f in self.iceflow_mapping["fieldin"]
                ],
                axis=-1,
            ),
            axis=0,
        )

        # Get the ice flow after applying the NN
#        Y = self.iceflow_model.predict_on_batch(X)
        Y = self.iceflow_model(X)

        # Appplying scaling, and update variables
        Ny, Nx = self.thk.shape
        for kk, f in enumerate(self.iceflow_mapping["fieldout"]):
            vars(self)[f].assign(
                tf.where(self.thk > 0, Y[0, :Ny, :Nx, kk], 0)
                * self.iceflow_fieldbounds[f]
            )

        # If requested, the speeds are artifically upper-bounded
        if self.config.force_max_velbar > 0:

            self.velbar_mag = self.getmag(self.ubar, self.vbar)

            self.ubar.assign(
                tf.where(
                    self.velbar_mag >= self.config.force_max_velbar,
                    self.config.force_max_velbar * (self.ubar / self.velbar_mag),
                    self.ubar,
                )
            )
            self.vbar.assign(
                tf.where(
                    self.velbar_mag >= self.config.force_max_velbar,
                    self.config.force_max_velbar * (self.vbar / self.velbar_mag),
                    self.vbar,
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
            help="Update the climate each X years (default: 1)",
        )
        self.parser.add_argument(
            "--type_climate", 
            type=str, default="", 
            help="This keywork serves to identify & call the climate forcing. If an empty string, this function is not called (Default: "")",
        )

    def update_climate(self):
        """
        This function serves to define a climate forcing (e.g. monthly temperature and precipitation fields) to be used for the surface mass balance model (e.g. accumulation/melt PDD-like model). No climate forcing is provided with IGM is this is case-dependent. Check at the aletsch-1880-21000 example.
        """

        if len(self.config.type_climate) > 0:

            if not hasattr(self, "already_called_update_climate"):

                getattr(self, "load_climate_data_" + self.config.type_climate)()
                self.tlast_clim = -1.0e5000
                self.tcomp["Climate"] = []
                self.already_called_update_climate = True

            new_clim_needed = (
                self.t.numpy() - self.tlast_clim
            ) >= self.config.clim_update_freq

            if new_clim_needed:

                if self.config.verbosity == 1:
                    print("Construct climate at time : ", self.t)

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
            help="This keywork permits to identify the type of mass balance model, can be: zero, simple, nn or given (Default: simple)",
        )
        self.parser.add_argument(
            "--mb_scaling", 
            type=float, 
            default=1.0, 
            help="The paramter permit to make a simple mass balance scaling"
        )
        self.parser.add_argument(
            "--mb_simple_file",
            type=str,
            default="mb_simple_param.txt",
            help="Name of the imput file for the simple mass balance model",
        )
        self.parser.add_argument(
            "--smb_model_lib_path",
            type=str,
            default="../../model-lib/smb_meteoswissglamos",
            help="Model directory in case the smb model in use is 'nn', i.e. neural network",
        )

    def init_smb_simple(self):
        """
        Read parameters of simple mass balance from an ascii file
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
        Implement Mass balance 'simple' parametrized by ELA, ablation and accumulation gradients, and max acuumulation
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

    def init_smb_nn(self):
        """
        initialize smb Neural Network
        """

        dirpath = os.path.join(self.config.smb_model_lib_path, str(int(self.dx)))

        assert os.path.isdir(dirpath)

        fieldin, fieldout, fieldbounds = self.read_fields_and_bounds(dirpath)

        self.read_fields_and_bounds(dirpath)

        self.smb_mapping = {}
        self.smb_mapping["fieldin"] = fieldin
        self.smb_mapping["fieldout"] = fieldout
        self.smb_fieldbounds = fieldbounds

        self.smb_model = tf.keras.models.load_model(os.path.join(dirpath, "model.h5"))

    def update_smb_nn(self):
        """
        function update the smb using the neural network
        """

        # this is not a nice implementation, but for now, it does the job
        self.mask = tf.ones_like(self.thk)
        for i in range(12):
            vars(self)["air_temp_" + str(i)] = self.air_temp[i]
            vars(self)["precipitation_" + str(i)] = self.precipitation[i]

        X = tf.expand_dims(
            tf.stack(
                [
                    vars(self)[f] / self.smb_fieldbounds[f]
                    for f in self.smb_mapping["fieldin"]
                ],
                axis=-1,
            ),
            axis=0,
        )

        Y = self.smb_model.predict_on_batch(X)

        # this will return the smb, the only output of the smb nn emulator
        for kk, f in enumerate(self.smb_mapping["fieldout"]):
            vars(self)[f].assign(Y[0, :, :, kk] * self.smb_fieldbounds[f])

    def update_smb(self):
        """
        This function permits to choose between several surface mass balance models:
                
        * A very simple mass balance model based on a few parameters (ELA, ...), whose parameters are defined in file glacier.config.mb_simple_file. This surface mass balance is provided with IGM.
                
        * Users can build their own mass balance routine, and possibly combine them with a climate routine. E.g. in the aletsch-1880-21000 example, both climate and surface mass balance models were customized to implement i) the computation of daily temperature and precipitation 2D fields ii) an accumulation/melt model (PDD-like) that takes the climate input, and transforms them into effective surface mass balance.
                
        * Surface Mass balance can be given in the form of a neural network, which predicts surface mass balance from temperature and precipitation fields. As an illustration, I have trained a Neural Network from climate and mass balance data from glaciers in the Alps using the [Deep Learning Emulator](https://github.com/jouvetg/dle). To try it, check the example aletsch-1880-2100. Note that this is highly experimental considering that so far i) the training dataset is small ii) no assessment was done.
        """

        if not hasattr(self, "already_called_update_smb"):
            self.tlast_mb = -1.0e5000
            self.tcomp["Mass balance"] = []
            if len(self.config.type_mass_balance) > 0:
                if hasattr(self, "init_smb_" + self.config.type_mass_balance):
                    getattr(self, "init_smb_" + self.config.type_mass_balance)()
            self.already_called_update_smb = True

        if ((self.t.numpy() - self.tlast_mb) >= self.config.mb_update_freq):

            if self.config.verbosity == 1:
                print("Construct mass balance at time : ", self.t.numpy())

            self.tcomp["Mass balance"].append(time.time())

            if len(self.config.type_mass_balance) > 0:
                getattr(self, "update_smb_" + self.config.type_mass_balance)()
            else:
                self.smb.assign(tf.zeros_like(self.topg))

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
    #                              BASAL EROSION
    ####################################################################################
    ####################################################################################
    ####################################################################################

    def read_config_param_erosion(self):

        self.parser.add_argument(
            "--erosion_include",
            type=str2bool,
            default=False,
            help="Include a model for bedrock erosion (Default: False)",
        )
        self.parser.add_argument(
            "--erosion_cst",
            type=float,
            default=2.7 * 10 ** (-7),
            help="Erosion multiplicative factor, here taken from Herman, F. et al. Erosion by an Alpine glacier. Science 350, 193–195 (2015)",
        )
        self.parser.add_argument(
            "--erosion_exp",
            type=float,
            default=2,
            help="Erosion exponent factor, here taken from Herman, F. et al. Erosion by an Alpine glacier. Science 350, 193–195 (2015)",
        )
        self.parser.add_argument(
            "--erosion_update_freq",
            type=float,
            default=100,
            help="Update the erosion only each X years (Default: 100)",
        )

        self.parser.add_argument(
            "--uplift_include",
            type=str2bool,
            default=False,
            help="Include a model with constant bedrock uplift",
        )
        self.parser.add_argument(
            "--uplift_rate", 
            type=float, 
            default=0.002, 
            help="Uplift rate in m/y (default 2 mm/y)",
        )
        self.parser.add_argument(
            "--uplift_update_freq",
            type=float,
            default=100,
            help="Update the uplift only each X years (Default: 100 years)",
        )

    def update_topg(self):
        """
        This function implements change in basal topography (glacial erosion or uplift). Setting glacier.config.erosion_include=True, the bedrock is updated (each glacier.config.erosion_update_freq years) assuming the erosion rate to be proportional (parameter glacier.config.erosion_cst) to a power (parameter glacier.config.erosion_exp) of the sliding velocity magnitude. By default, we use the parameters from Herman, F. et al. Erosion by an Alpine glacier. Science 350, 193–195 (2015). Check at the function glacier.update_topg() for more details on the implementation of glacial erosion in IGM. Setting glacier.config.uplift_include=True will allow to include an uplift defined by glacier.config.uplift_rate.
        """

        if (self.config.erosion_include)|(self.config.uplift_include):

            if not hasattr(self, "already_called_update_topg"):
                self.tlast_erosion = self.config.tstart
                self.tlast_uplift = self.config.tstart
                self.dtopgdt = tf.Variable(tf.zeros_like(self.thk))
                self.tcomp["Erosion"] = []
                self.tcomp["Uplift"] = []
                self.already_called_update_topg = True    

        if self.config.erosion_include:

            if (self.t.numpy() - self.tlast_erosion) >= self.config.erosion_update_freq:

                if self.config.verbosity == 1:
                    print("Erode bedrock at time : ", self.t.numpy())

                self.tcomp["Erosion"].append(time.time())

                self.velbase_mag = self.getmag(self.uvelbase, self.vvelbase)

                # apply erosion law, erosion rate is proportional to a power of basal sliding speed
                self.dtopgdt.assign(
                    self.config.erosion_cst
                    * (self.velbase_mag ** self.config.erosion_exp)
                )

                self.topg.assign(
                    self.topg - (self.t.numpy() - self.tlast_erosion) * self.dtopgdt
                )

                print("max erosion is :", np.max(np.abs(self.dtopgdt)))

                self.usurf.assign(self.topg + self.thk)

                self.tlast_erosion = self.t.numpy()

                self.tcomp["Erosion"][-1] -= time.time()
                self.tcomp["Erosion"][-1] *= -1

        if self.config.uplift_include:

            if (self.t.numpy() - self.tlast_uplift) >= self.config.uplift_update_freq:

                if self.config.verbosity == 1:
                    print("Uplift bedrock at time : ", self.t.numpy())

                self.tcomp["Uplift"].append(time.time())

                # apply constant uplift rate
                self.topg.assign(
                    self.topg
                    + self.config.uplift_rate * (self.t.numpy() - self.tlast_uplift)
                )

                self.usurf.assign(self.topg + self.thk)

                self.tlast_uplift = self.t.numpy()

                self.tcomp["Uplift"][-1] -= time.time()
                self.tcomp["Uplift"][-1] *= -1

    ####################################################################################
    ####################################################################################
    ####################################################################################
    #                              TRANSPORT
    ####################################################################################
    ####################################################################################
    ####################################################################################

    def update_thk(self):
        """
        This function solves the mass conservation of ice to update the thickness from ice flow and surface mass balance. The mass conservation equation is solved using an explicit first-order upwind finite-volume scheme on a regular 2D grid with constant cell spacing in any direction. The discretization and the approximation of the flux divergence is described [here](https://github.com/jouvetg/igm/blob/main/fig/transp-igm.jpg). With this scheme mass of ice is allowed to move from cell to cell (where thickness and velocities are defined) from edge-defined fluxes (inferred from depth-averaged velocities, and ice thickness in upwind direction). The resulting scheme is mass conservative and parallelizable (because fully explicit). However, it is subject to a CFL condition. This means that the time step (defined in glacier.update_t_dt()) is controlled by parameter glacier.config.cfl, which is the maximum number of cells crossed in one iteration (this parameter cannot exceed one).
        """

        if not hasattr(self, "already_called_update_icethickness"):
            self.tcomp["Transport"] = []
            self.already_called_update_icethickness = True

        else:
            if self.config.verbosity == 1:
                print("Ice thickness equation at time : ", self.t.numpy())

            self.tcomp["Transport"].append(time.time())

            # compute the divergence of the flux
            self.divflux = self.compute_divflux(
                self.ubar, self.vbar, self.thk, self.dx, self.dx
            )

            # Forward Euler with projection to keep ice thickness non-negative
            self.thk.assign(
                tf.maximum(self.thk + self.dt * (self.smb - self.divflux), 0)
            )

            # update ice surface accordingly
            self.usurf.assign(self.topg + self.thk)

            # update gradients of the surface (slopes)
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
        u = tf.concat(
            [u[:, 0:1], 0.5 * (u[:, :-1] + u[:, 1:]), u[:, -1:]], 1
        )  # has shape (ny,nx+1)
        v = tf.concat(
            [v[0:1, :], 0.5 * (v[:-1, :] + v[1:, :]), v[-1:, :]], 0
        )  # has shape (ny+1,nx)

        # Extend h with constant value at the domain boundaries
        Hx = tf.pad(h, [[0, 0], [1, 1]], "CONSTANT")  # has shape (ny,nx+2)
        Hy = tf.pad(h, [[1, 1], [0, 0]], "CONSTANT")  # has shape (ny+2,nx)

        ## Compute fluxes by selcting the upwind quantities
        Qx = u * tf.where(u > 0, Hx[:, :-1], Hx[:, 1:])  # has shape (ny,nx+1)
        Qy = v * tf.where(v > 0, Hy[:-1, :], Hy[1:, :])  # has shape (ny+1,nx)

        ## Computation of the divergence, final shape is (ny,nx)
        return (Qx[:, 1:] - Qx[:, :-1]) / dx + (Qy[1:, :] - Qy[:-1, :]) / dy

    ####################################################################################
    ####################################################################################
    ####################################################################################
    #                               TRACKING
    ####################################################################################
    ####################################################################################
    ####################################################################################

    def read_config_param_tracking(self):

        self.parser.add_argument(
            "--tracking_method",
            type=str,
            default='3d',
            help="Method for tracking particles (3d or simple)",
        )

        self.parser.add_argument(
            "--frequency_seeding",
            type=int,
            default=10,
            help="Frequency of seeding (default: 10)",
        )

        self.parser.add_argument(
            "--density_seeding", 
            type=int, default=0.2, 
            help="Density of seeding (default: 0.2)",
        )

    def seeding_particles(self):
        """
        here we define (xpos,ypos) the horiz coordinate of tracked particles
        and rhpos is the relative position in the ice column (scaled bwt 0 and 1)
                
        here we seed only the accum. area (a bit more), where there is 
        significant ice, and in some points of a regular grid self.gridseed
        (density defined by density_seeding)
        
        """

        #        This will serve to remove imobile particles, but it is not active yet.

        #        indices = tf.expand_dims( tf.concat(
        #                       [tf.expand_dims((self.ypos - self.y[0]) / self.dx, axis=-1),
        #                        tf.expand_dims((self.xpos - self.x[0]) / self.dx, axis=-1)],
        #                       axis=-1 ), axis=0)

        #        import tensorflow_addons as tfa

        #        thk = tfa.image.interpolate_bilinear(
        #                    tf.expand_dims(tf.expand_dims(self.thk, axis=0), axis=-1),
        #                    indices,indexing="ij",      )[0, :, 0]

        #        J = (thk>1)

        I = (self.thk > 10) & (self.smb > -2) & self.gridseed # seed where thk>10, smb>-2, on a coarse grid
        self.nxpos  = self.X[I]
        self.nypos  = self.Y[I]
        self.nzpos  = self.usurf[I]
        self.nrhpos = tf.ones_like(self.X[I])
        self.nwpos  = tf.ones_like(self.X[I])

    def update_particles(self):
        """
        IGM includes a particle tracking routine, which can compute a large number of trajectories (as it is implemented with TensorFlow to run in parallel) in live time during the forward model run. The routine produces some seeding of particles (by default in the accumulation area at regular intervals), and computes the time trajectory of the resulting particle in time advected by the velocity field in 3D. There are currently 2 implementations:
            
        * 'simple', Horizontal and vertical directions are treated differently: i) In the horizontal plan, particles are advected with the horizontal velocity field (interpolated bi-linearly). The positions are recorded in vector (glacier.xpos,glacier.ypos). ii) In the vertical direction, particles are tracked along the ice column scaled between 0 and 1 (0 at the bed, 1 at the top surface). The relative position along the ice column is recorded in vector glacier.rhpos (same dimension as glacier.xpos and iglaciergm.ypos). Particles are always initialized at 1 (assumed to be on the surface). The evolution of the particle within the ice column through time is computed according to the surface mass balance: the particle deepens when the surface mass balance is positive (then igm.rhpos decreases), and re-emerge when the surface mass balance is negative.  
    
        * '3d', The vertical velocity is reconsructed by integrating the divergence of the horizontal velocity, this permits in turn to perform 3D particle tracking. self.zpos is the z- position within the ice.
        
        Note that in both case, the velocity in the ice layer is reconstructed from bottom and surface one assuming 4rth order polynomial profile (SIA-like)
        
        To include this feature, make sure:
        
        * To adapt the seeding to your need. You may keep the default seeding in the accumulation area setting the seeding frequency with igm.config.frequency_seeding and the seeding density glacier.config.density_seeding. Alternatively, you may define your own seeding strategy (e.g. seeding close to rock walls/nunataks). To do so, you may redefine the function seeding_particles.
        
        * At each time step, the weight of surface debris contains in each cell the 2D horizontal grid is computed, and stored in variable igm.weight_particles.
        
        * You may visualize the moving particles in update_plot(). 
        
        * Particles trajectories are saved in folder 'trajectory', and are organized by seeding times.
        
        Check at the example aletsch-1880-2100 for an example of particle tracking.
        """

        import tensorflow_addons as tfa

        if self.config.verbosity == 1:
            print("Update TRACKING at time : ", self.t)

        if not hasattr(self, "already_called_update_tracking"):
            self.already_called_update_tracking = True
            self.tlast_seeding = -1.0e5000
            self.tcomp["Tracking"] = []

            # initialize trajectories
            self.xpos = tf.Variable([])
            self.ypos = tf.Variable([])
            self.zpos = tf.Variable([])
            self.rhpos = tf.Variable([])
            self.wpos = tf.Variable([]) # this is to give a weight to the particle

            # build the gridseed
            self.gridseed = np.zeros_like(self.thk) == 1
            rr = int(1.0 / self.config.density_seeding)
            self.gridseed[::rr, ::rr] = True

            # create trajectory folder to store outputs
            directory = os.path.join(self.config.working_dir, "trajectories")
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.mkdir(directory)
            
            os.system('echo rm -r '+ os.path.join(self.config.working_dir, "trajectories") + ' >> clean.sh')
  
        if (self.t.numpy() - self.tlast_seeding) >= self.config.frequency_seeding:
            self.seeding_particles()

            # merge the new seeding points with the former ones
            self.xpos = tf.Variable(tf.concat([self.xpos, self.nxpos], axis=-1))
            self.ypos = tf.Variable(tf.concat([self.ypos, self.nypos], axis=-1))
            self.zpos = tf.Variable(tf.concat([self.zpos, self.nzpos], axis=-1))
            self.rhpos = tf.Variable(tf.concat([self.rhpos, self.nrhpos], axis=-1))
            self.wpos = tf.Variable(tf.concat([self.wpos, self.nwpos], axis=-1))

            self.tlast_seeding = self.t.numpy() 

        self.tcomp["Tracking"].append(time.time())
        
        self.update_write_trajectories()

        # find the indices of trajectories
        # these indicies are real values to permit 2D interpolations
        i = (self.xpos - self.x[0]) / self.dx
        j = (self.ypos - self.y[0]) / self.dx

        indices = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(j, axis=-1), tf.expand_dims(i, axis=-1)], axis=-1
            ),
            axis=0,
        )

        uvelbase = tfa.image.interpolate_bilinear(
            tf.expand_dims(tf.expand_dims(self.uvelbase, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]

        vvelbase = tfa.image.interpolate_bilinear(
            tf.expand_dims(tf.expand_dims(self.vvelbase, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]

        uvelsurf = tfa.image.interpolate_bilinear(
            tf.expand_dims(tf.expand_dims(self.uvelsurf, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]

        vvelsurf = tfa.image.interpolate_bilinear(
            tf.expand_dims(tf.expand_dims(self.vvelsurf, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]

        othk = tfa.image.interpolate_bilinear(
            tf.expand_dims(tf.expand_dims(self.thk, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
        
        topg = tfa.image.interpolate_bilinear(
            tf.expand_dims(tf.expand_dims(self.topg, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
         
        smb = tfa.image.interpolate_bilinear(
            tf.expand_dims(tf.expand_dims(self.smb, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
        
        if self.config.tracking_method=='simple':

            nthk = othk + smb * self.dt  # new ice thicnkess after smb update

            # adjust the relative height within the ice column with smb
            self.rhpos.assign(
                tf.where(
                    nthk > 0.1, tf.clip_by_value(self.rhpos * othk / nthk, 0, 1), 1
                )
            )
            
            uvel = uvelbase + (uvelsurf - uvelbase) * (
                1 - (1 - self.rhpos) ** 4
            )  # SIA-like
            vvel = vvelbase + (vvelsurf - vvelbase) * (
                1 - (1 - self.rhpos) ** 4
            )  # SIA-like

            self.xpos.assign(self.xpos + self.dt * uvel)  # forward euler
            self.ypos.assign(self.ypos + self.dt * vvel)  # forward euler 
            
            self.zpos.assign( topg + nthk * self.rhpos  )
            
        elif self.config.tracking_method=='3d':
            
# This was a test of smoothing the surface topography to regaluraze the vertical velocitiy.                
#                import tensorflow_addons as tfa
#                susurf = tfa.image.gaussian_filter2d(self.usurf, sigma=5, filter_shape=5, padding="CONSTANT")
#                stopg  = tfa.image.gaussian_filter2d(self.topg , sigma=3, filter_shape=5, padding="CONSTANT")
            
            slopsurfx, slopsurfy = self.compute_gradient_tf(self.usurf, self.dx, self.dx)
            sloptopgx, sloptopgy = self.compute_gradient_tf(self.topg, self.dx, self.dx)
    
            self.divflux = self.compute_divflux(self.ubar, self.vbar, self.thk, self.dx, self.dx)
     
            # the vertical velocity is the scalar product of horizont. velo and bedrock gradient
            self.wvelbase =  self.uvelbase * sloptopgx + self.vvelbase * sloptopgy
            # Using rules of derivative the surface vertical velocity can be found from the
            # divergence of the flux considering that the ice 3d velocity is divergence-free.
            self.wvelsurf =  self.uvelsurf * slopsurfx + self.vvelsurf * slopsurfy - self.divflux 
            
            wvelbase = tfa.image.interpolate_bilinear(
                tf.expand_dims(tf.expand_dims(self.wvelbase, axis=0), axis=-1),
                indices,
                indexing="ij",
            )[0, :, 0]
            
            wvelsurf = tfa.image.interpolate_bilinear(
                tf.expand_dims(tf.expand_dims(self.wvelsurf, axis=0), axis=-1),
                indices,
                indexing="ij",
            )[0, :, 0]
            
#           print('at the surface? : ',all(self.zpos == topg+othk))

            # make sure the particle remian withi the ice body
            self.zpos.assign( tf.clip_by_value( self.zpos , topg, topg+othk) )
            
            # get the relative height
            self.rhpos.assign( tf.where( othk > 0.1, (self.zpos - topg)/othk , 1 ) )
             
            uvel = uvelbase + (uvelsurf - uvelbase) * ( 1 - (1 - self.rhpos) ** 4)  # SIA-like
            vvel = vvelbase + (vvelsurf - vvelbase) * ( 1 - (1 - self.rhpos) ** 4)  # SIA-like
            wvel = wvelbase + (wvelsurf - wvelbase) * ( 1 - (1 - self.rhpos) ** 4)  # SIA-like
               
            self.xpos.assign(self.xpos + self.dt * uvel)  # forward euler
            self.ypos.assign(self.ypos + self.dt * vvel)  # forward euler 
            self.zpos.assign(self.zpos + self.dt * wvel)  # forward euler 
            
            # make sur the particle remains in the horiz. comp. domain
            self.xpos.assign(tf.clip_by_value(self.xpos,self.x[0],self.x[-1]))
            self.ypos.assign(tf.clip_by_value(self.ypos,self.y[0],self.y[-1]))

        indices = tf.concat(
            [
                tf.expand_dims(tf.cast(j, dtype="int32"), axis=-1),
                tf.expand_dims(tf.cast(i, dtype="int32"), axis=-1),
            ],
            axis=-1,
        )
        updates = tf.cast(tf.where(self.rhpos == 1, self.wpos, 0), dtype="float32")
        
        # this computes the sum of the weight of particles on a 2D grid
        self.weight_particles = tf.tensor_scatter_nd_add(
            tf.zeros_like(self.thk), indices, updates
        )

        self.tcomp["Tracking"][-1] -= time.time()
        self.tcomp["Tracking"][-1] *= -1
            
    def update_write_trajectories(self):

        if self.saveresult:
            
            if not hasattr(self, "already_called_update_write_trajectories"):
                self.already_called_update_write_trajectories = True
                ftt = os.path.join(self.config.working_dir, "trajectories", 'topg.csv') 
                array = tf.transpose(tf.stack([self.X[self.X>0],self.Y[self.X>0],self.topg[self.X>0]]))
                np.savetxt(ftt, array , delimiter=',', fmt="%.2f", header='x,y,z')
             
            f = os.path.join(self.config.working_dir, "trajectories", 'traj-'+'{:06d}'.format(int(self.t.numpy()))+'.csv') 
            ft = os.path.join(self.config.working_dir, "trajectories", 'time.dat') 
            ID = tf.cast(tf.range(self.xpos.shape[0]),dtype='float32')
            array = tf.transpose(tf.stack([ID,self.xpos,self.ypos,self.zpos,self.rhpos],axis=0))
            np.savetxt(f, array , delimiter=',', fmt="%.2f", header='Id,x,y,z,rh')
            with open(ft, "a") as f:
                print(self.t.numpy(), file=f )  
                
            ftt = os.path.join(self.config.working_dir, "trajectories", 'usurf-'+'{:06d}'.format(int(self.t.numpy()))+'.csv') 
            array = tf.transpose(tf.stack([self.X[self.X>1],self.Y[self.X>1],self.usurf[self.X>1]]))
            np.savetxt(ftt, array , delimiter=',', fmt="%.2f", header='x,y,z')

    # def update_write_trajectories_old(self):

    #     if self.saveresult:

    #         ft = os.path.join(self.config.working_dir, "trajectories", 't.dat')
    #         fx = os.path.join(self.config.working_dir, "trajectories", 'x.dat')
    #         fy = os.path.join(self.config.working_dir, "trajectories", 'y.dat')
    #         fz = os.path.join(self.config.working_dir, "trajectories", 'z.dat')
    #         fr = os.path.join(self.config.working_dir, "trajectories", 'r.dat')                        

    #         with open(ft, "a") as f:
    #             print(self.t.numpy(), file=f )    
                
    #         with open(fx, "a") as f:
    #             print(*list(self.xpos.numpy()), file=f )      
 
    #         with open(fy, "a") as f:
    #             print(*list(self.ypos.numpy()), file=f )    
                
    #         with open(fz, "a") as f:
    #             print(*list(self.zpos.numpy()), file=f )    
                
    #         with open(fr, "a") as f:
    #             print(*list(self.rhpos.numpy()), file=f )  
    
    ####################################################################################
    ####################################################################################
    ####################################################################################
    #                               THK SMOOTHING
    ####################################################################################
    ####################################################################################
    ####################################################################################
    
    def read_config_param_smoothing_thk(self):

        self.parser.add_argument(
            "--smoothing_thk_filter_shape",
            type=int,
            default=3,
            help="smoothing_thk_filter_shape",
        )
        self.parser.add_argument(
            "--smoothing_thk_sigma",
            type=float,
            default=3,
            help="smoothing_thk_sigma",
        )
        self.parser.add_argument(
            "--smoothing_thk_update_freq",
            type=float,
            default=100,
            help="Update the smoothing thk only each X years (Default: 1)",
        )
 
    def update_smoothing_thk(self):
        """
        This function permits to smooth the ice thickness at some intervals
        """

        if not hasattr(self, "already_called_smoothing_thk"):
            self.tlast_smoothing_thk         = -1.0e5000  
            self.already_called_smoothing_thk = True
            
        if ((self.t.numpy() - self.tlast_smoothing_thk) >= self.config.smoothing_thk_update_freq):
         
                import tensorflow_addons as tfa
                 
                self.thk.assign( tfa.image.gaussian_filter2d(self.thk, 
                                                       sigma=self.config.smoothing_thk_sigma, 
                                                       filter_shape=self.config.smoothing_thk_filter_shape,
                                                       padding="CONSTANT") )
                
                self.tlast_smoothing_thk = self.t.numpy()

    ####################################################################################
    ####################################################################################
    ####################################################################################
    #                               3DVEL
    ####################################################################################
    ####################################################################################
    ####################################################################################

    def read_config_param_3dvel(self):

        self.parser.add_argument(
            "--dz",
            type=int,
            default=20,
            help="Vertical discretization constant spacing",
        )

        self.parser.add_argument(
            "--maxthk", type=float, default=1000.0, help="Vertical maximum thickness"
        )

    def init_3dvel(self):

        self.height = np.arange(
            0, self.config.maxthk + 1, self.config.dz
        )  # height with constant dz
        self.ddz = self.height[1:] - self.height[:-1]

        self.nz = tf.Variable(
            tf.zeros((self.thk.shape[0], self.thk.shape[1]), dtype="int32")
        )
        self.depth = tf.Variable(
            tf.zeros(
                (self.height.shape[0], self.thk.shape[0], self.thk.shape[1]),
                dtype="float32",
            )
        )
        self.dz = tf.Variable(
            tf.zeros(
                (self.height.shape[0] - 1, self.thk.shape[0], self.thk.shape[1]),
                dtype="float32",
            )
        )

        self.U = tf.Variable(tf.zeros_like(self.depth))  # x-vel component
        self.V = tf.Variable(tf.zeros_like(self.depth))  # y-vel component
        self.W = tf.Variable(tf.zeros_like(self.depth))  # z-vel component

        self.tcomp["3d Vel"] = []

    @tf.function()
    def update_vert_disc_tf(self):

        # nz is the index of the first node above the ice surface
        nz = tf.ones(self.thk.shape, "int32") * (self.height.shape[0] - 1)
        for k in range(self.height.shape[0] - 1, -1, -1):
            nz = tf.where(self.height[k] > self.thk, k, nz)
        self.nz.assign(tf.where(self.thk >= self.ddz[0], nz, 1))

        # depth is the depth of ice at any grid point, otherwise it is zero
        depth = []
        for k in range(0, self.height.shape[0]):
            depth.append(
                tf.where(
                    (self.thk >= self.ddz[0]) & (k < self.nz),
                    self.thk - self.height[k],
                    0.0,
                )
            )
        self.depth.assign(tf.stack(depth, axis=0))

        # dz is the  vertical spacing,
        dz = []
        for k in range(0, self.height.shape[0] - 1):
            dz.append(tf.ones_like(self.thk) * self.ddz[k])
        self.dz.assign(tf.stack(dz, axis=0))

    @tf.function()
    def update_reconstruct_3dvel_tf(self):

        # Reconstruct the horizontal velocity field from basal and surface velocity assuming a SIA-like profile
        fshear = []
        U = []
        V = []
        W = []

        fshear.append(tf.zeros_like(self.thk))

        for k in range(1, self.height.shape[0]):
            fshear.append(
                fshear[-1]
                + tf.where(k < self.nz, self.dz[k - 1], 0.0) * (self.depth[k] ** 3)
            )

        norm = fshear[-1]

        for k in range(0, self.height.shape[0]):
            fshear[k] /= tf.where(self.nz > 1, norm, 1.0)

        for k in range(0, self.height.shape[0]):
            U.append(self.uvelbase + fshear[k] * (self.uvelsurf - self.uvelbase))
            V.append(self.vvelbase + fshear[k] * (self.vvelsurf - self.vvelbase))
            W.append(self.vvelbase + fshear[k] * (self.wvelsurf - self.wvelbase))

        self.U.assign(tf.stack(U, axis=0))
        self.V.assign(tf.stack(V, axis=0))
        self.W.assign(tf.stack(W, axis=0))

        # Ui = tf.pad(self.U, [[0, 0], [0, 0], [1, 1]], "SYMMETRIC")
        # Vj = tf.pad(self.V, [[0, 0], [1, 1], [0, 0]], "SYMMETRIC")

        ######### THis methods reconstruct the vertical velocity using divflux,
        ######### and assuming a SIA-like profile like x- and y- components

        # slopsurfx, slopsurfy = self.compute_gradient_tf(self.usurf, self.dx, self.dx)
        # sloptopgx, sloptopgy = self.compute_gradient_tf(self.topg, self.dx, self.dx)

        # divflux = self.compute_divflux(self.ubar, self.vbar, self.thk, self.dx, self.dx)

        # self.wvelbase = self.uvelbase * sloptopgx + self.vvelbase * sloptopgy
        # self.wvelsurf = -divflux + self.uvelsurf * slopsurfx + self.vvelsurf * slopsurfy

        # for k in range(0, self.height.shape[0]):
        #     W.append(self.wvelbase + fshear[k] * (self.wvelsurf - self.wvelbase))

        # self.W.assign(tf.stack(W, axis=0))

        ### This methods integrates the imcompressiblity conditoons

        # W.append( tf.zeros_like(self.thk) )

        # for k in range(1,self.height.shape[0]):
        #     W.append( W[-1] + tf.where( k<self.nz, \
        #                                 self.dz[k-1] * ( - (Ui[k-1,:, 2:] - Ui[k-1,:,:-2]) / (2*self.dx) \
        #                                                  - (Vj[k-1,2:, :] - Vj[k-1,:-2,:]) / (2*self.dx) ), \
        #                                 0.0 )
        #             )

        # self.W.assign( tf.stack(W,axis=0) )

    def update_3dvel(self):

        if self.config.verbosity == 1:
            print("update_3dvel ")

        self.tcomp["3d Vel"].append(time.time())

        self.update_vert_disc_tf()

        self.update_reconstruct_3dvel_tf()

        self.update_ncdf_3d_ex()

        self.tcomp["3d Vel"][-1] -= time.time()
        self.tcomp["3d Vel"][-1] *= -1

    def update_ncdf_3d_ex(self):
        """
        Initialize  and write the ncdf output file
        """

        if self.saveresult:

            if self.it == 0:

                if self.config.verbosity == 1:
                    print("Initialize NCDF output Files")

                nc = Dataset(
                    os.path.join(self.config.working_dir, "ex3d.nc"),
                    "w",
                    format="NETCDF4",
                )

                nc.createDimension("time", None)
                E = nc.createVariable("time", np.dtype("float32").char, ("time",))
                E.units = "yr"
                E.long_name = "time"
                E.axis = "T"
                E[0] = self.t.numpy()

                nc.createDimension("h", self.height.shape[0])
                E = nc.createVariable("h", np.dtype("float32").char, ("h",))
                E.units = "m"
                E.long_name = "h"
                E.standard_name = "h"
                E.axis = "H"
                E[:] = self.height

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

                E = nc.createVariable(
                    "topg", np.dtype("float32").char, ("time", "y", "x")
                )
                E.units = "m"
                E.long_name = "topg"
                E.standard_name = "topg"
                E[0, :, :] = self.topg.numpy()

                E = nc.createVariable(
                    "usurf", np.dtype("float32").char, ("time", "y", "x")
                )
                E.units = "m"
                E.long_name = "usurf"
                E.standard_name = "usurf"
                E[0, :, :] = self.usurf.numpy()

                E = nc.createVariable(
                    "U", np.dtype("float32").char, ("time", "h", "y", "x")
                )
                E.units = "m/y"
                E.long_name = "U"
                E.standard_name = "U"
                E[0, :, :, :] = self.U.numpy()

                E = nc.createVariable(
                    "V", np.dtype("float32").char, ("time", "h", "y", "x")
                )
                E.units = "m/y"
                E.long_name = "V"
                E.standard_name = "V"
                E[0, :, :, :] = self.V.numpy()

                E = nc.createVariable(
                    "W", np.dtype("float32").char, ("time", "h", "y", "x")
                )
                E.units = "m/y"
                E.long_name = "W"
                E.standard_name = "W"
                E[0, :, :, :] = self.W.numpy()

                nc.close()

            else:

                nc = Dataset(
                    os.path.join(self.config.working_dir, "ex3d.nc"),
                    "a",
                    format="NETCDF4",
                )

                d = nc.variables["time"][:].shape[0]
                nc.variables["time"][d] = self.t.numpy()
                nc.variables["U"][d, :, :, :] = self.U.numpy()
                nc.variables["V"][d, :, :, :] = self.V.numpy()
                nc.variables["W"][d, :, :, :] = self.W.numpy()
                nc.variables["usurf"][d, :, :] = self.usurf.numpy()
                nc.variables["topg"][d, :, :] = self.topg.numpy()

                nc.close()

    ####################################################################################
    ####################################################################################
    ####################################################################################
    #                               OPTI
    ####################################################################################
    ####################################################################################
    ####################################################################################

    def read_config_param_optimize(self):

        # OPTIMIZATION PARAMETERS
        self.parser.add_argument(
            "--optimize",
            type=str2bool,
            default=False,
            help="Do the data ssimilation (inverse modelling) prior forward modelling (Default: False)",
        )
        self.parser.add_argument(
            "--opti_vars_to_save",
            type=list,
            default=[
                "topg",
                "usurf",
                "thk",
                "strflowctrl",
                "arrhenius",
                "slidingco",
                "velsurf_mag",
                "velsurfobs_mag",
                "divflux",
            ],
            help="List of variables to be recorded in the ncdef file",
        )
        self.parser.add_argument(
            "--observation_file",
            type=str,
            default="observation.nc",
            help="Observation file contains the 2D data observations fields (thkobs, usurfobs, uvelsurfobs, ....) ",
        )
        self.parser.add_argument(
            "--thk_profiles_file",
            type=str,
            default="",
            help="Provide ice thickness measurements, if empty string it will look for rasterized thk in the input data file",
        )
        self.parser.add_argument("--mode_opti", type=str, default="thkstrflowctrl")

        self.parser.add_argument(
            "--opti_thr_strflowctrl",
            type=float,
            default=78.0,
            help="threshold value for strflowctrl",
        )
        self.parser.add_argument(
            "--opti_init_zero_thk",
            type=str2bool,
            default="False",
            help="Initialize the optimization with zero ice thickness",
        )
        self.parser.add_argument(
            "--opti_regu_param_thk",
            type=float,
            default=10.0,
            help="Regularization weight for the ice thickness in the optimization",
        )
        self.parser.add_argument(
            "--opti_regu_param_strflowctrl",
            type=float,
            default=1.0,
            help="Regularization weight for the strflowctrl field in the optimization",
        )
        self.parser.add_argument(
            "--opti_smooth_anisotropy_factor",
            type=float,
            default=0.2,
            help="Smooth anisotropy factor for the ice thickness regularization in the optimization",
        )
        self.parser.add_argument(
            "--opti_convexity_weight",
            type=float,
            default=0.002,
            help="Convexity weight for the ice thickness regularization in the optimization",
        )
        self.parser.add_argument(
            "--opti_usurfobs_std",
            type=float,
            default=5.0,
            help="Confidence/STD of the top ice surface as input data for the optimization",
        )
        self.parser.add_argument(
            "--opti_strflowctrl_std",
            type=float,
            default=5.0,
            help="Confidence/STD of strflowctrl",
        )
        self.parser.add_argument(
            "--opti_velsurfobs_std",
            type=float,
            default=3.0,
            help="Confidence/STD of the surface ice velocities as input data for the optimization (if 0, velsurfobs_std field must be given)",
        )
        self.parser.add_argument(
            "--opti_thkobs_std",
            type=float,
            default=5.0,
            help="Confidence/STD of the ice thickness profiles (unless given)",
        )
        self.parser.add_argument(
            "--opti_divfluxobs_std",
            type=float,
            default=1.0,
            help="Confidence/STD of the flux divergence as input data for the optimization (if 0, divfluxobs_std field must be given)",
        )
        self.parser.add_argument(
            "--opti_control",
            type=list,
            default=["thk", "strflowctrl", "usurf"],
            help="List of optimized variables for the optimization",
        )
        self.parser.add_argument(
            "--opti_cost",
            type=list,
            default=["velsurf", "thk", "usurf", "divfluxfcz", "icemask"],
            help="List of cost components for the optimization",
        )
        self.parser.add_argument(
            "--opti_nbitmin",
            type=int,
            default=50,
            help="Min iterations for the optimization",
        )
        self.parser.add_argument(
            "--opti_nbitmax",
            type=int,
            default=1000,
            help="Max iterations for the optimization",
        )
        self.parser.add_argument(
            "--opti_step_size",
            type=float,
            default=0.001,
            help="Step size for the optimization",
        )
        self.parser.add_argument(
            "--opti_make_holes_in_data",
            type=int,
            default=0,
            help="This produces artifical holes in data, serve to test the robustness of the method to missing data",
        )
        self.parser.add_argument(
            "--opti_output_freq",
            type=int,
            default=50,
            help="Frequency of the output for the optimization",
        )
        self.parser.add_argument(
            "--geology_optimized_file",
            type=str,
            default="geology-optimized.nc",
            help="Geology input file",
        )

    def make_data_holes(self):
        """
        This serves to make holes in surface data (velocities) to test robusness of the optimization
        """

        mask_holes = np.zeros_like(self.thk.numpy(), dtype="int")

        if (self.config.opti_make_holes_in_data > 0) & (
            self.config.opti_make_holes_in_data < 99
        ):

            np.random.seed(seed=123)

            while (
                np.sum(mask_holes) / np.sum(self.icemaskobs > 0.5)
            ) < self.config.opti_make_holes_in_data * 0.01:

                j = np.random.randint(0, self.thk.shape[0])
                i = np.random.randint(0, self.thk.shape[1])

                if self.icemaskobs[j, i] > 0.5:
                    mask_holes[j, i] = 1

        elif self.config.opti_make_holes_in_data == 100:
            mask_holes = np.ones_like(self.thk.numpy(), dtype="int")

        elif self.config.opti_make_holes_in_data == 200:
            mask_holes[self.Y > np.mean(self.y)] = 1

        self.uvelsurfobs = tf.where(
            (self.icemaskobs == 1) & (mask_holes == 0), self.uvelsurfobs, np.nan
        )
        self.vvelsurfobs = tf.where(
            (self.icemaskobs == 1) & (mask_holes == 0), self.vvelsurfobs, np.nan
        )

    def load_thk_profiles(self):
        """
        load glacier ice thickness mesured profiles
        """

        pathprofile = os.path.join(
            self.config.working_dir, self.config.thk_profiles_file
        )

        if len(glob.glob(pathprofile)) > 0:

            vvv = []
            for g in glob.glob(pathprofile):
                vvv.append(np.loadtxt(g, dtype=np.float32))
            self.profile = np.concatenate(vvv, axis=0)

            # re-index profiles in case there were not properly indexed
            self.profile = [
                self.profile[np.floor(self.profile[:, 0]) == i + 1, :]
                for i in range(int(np.max(self.profile[:, 0])))
            ]

            # Remove empty List from List using list comprehension
            self.profile = [ele for ele in self.profile if len(ele) > 0]

            # compute the distance between points as first entry as it will be usefully later on
            for p in self.profile:
                p[:, 0] = np.insert(
                    np.cumsum(np.sqrt(np.diff(p[:, 1]) ** 2 + np.diff(p[:, 2]) ** 2)),
                    0,
                    0,
                )
        else:
            sys.exit("thk requested in opti, but file " + pathprofile + " is not found")

        points = np.stack([item[1:3] for sublist in self.profile for item in sublist])
        jiptprof = []
        for p in points:
            i = (p[0] - np.min(self.X)) / self.dx
            j = (p[1] - np.min(self.Y)) / self.dx
            jiptprof.append([j, i])
        self.jiptprof = tf.expand_dims(jiptprof, axis=0)

        target = np.stack([item[3] for sublist in self.profile for item in sublist])
        self.thktargetprof = tf.Variable(target.astype("float32"))

        target = np.stack([item[4] for sublist in self.profile for item in sublist])
        self.thktargetprof_std = tf.Variable(target.astype("float32"))

    def compute_flow_direction_for_anisotropic_smoothing(self):
        """
        compute_flow_direction_for_anisotropic_smoothing
        """

        uvelsurfobs = tf.where(tf.math.is_nan(self.uvelsurfobs), 0.0, self.uvelsurfobs)
        vvelsurfobs = tf.where(tf.math.is_nan(self.vvelsurfobs), 0.0, self.vvelsurfobs)

        self.flowdirx = (
            uvelsurfobs[1:, 1:]
            + uvelsurfobs[:-1, 1:]
            + uvelsurfobs[1:, :-1]
            + uvelsurfobs[:-1, :-1]
        ) / 4.0
        self.flowdiry = (
            vvelsurfobs[1:, 1:]
            + vvelsurfobs[:-1, 1:]
            + vvelsurfobs[1:, :-1]
            + vvelsurfobs[:-1, :-1]
        ) / 4.0

        from scipy.ndimage import gaussian_filter

        self.flowdirx = gaussian_filter(self.flowdirx, 3, mode="constant")
        self.flowdiry = gaussian_filter(self.flowdiry, 3, mode="constant")

        # Same as gaussian filter above but for tensorflow is (NOT TESTED)
        # import tensorflow_addons as tfa
        # self.flowdirx.assign( tfa.image.gaussian_filter2d( self.flowdirx , sigma=3, filter_shape=100, padding="CONSTANT") )

        self.flowdirx /= self.getmag(self.flowdirx, self.flowdiry)
        self.flowdiry /= self.getmag(self.flowdirx, self.flowdiry)

        self.flowdirx = tf.where(tf.math.is_nan(self.flowdirx), 0.0, self.flowdirx)
        self.flowdiry = tf.where(tf.math.is_nan(self.flowdiry), 0.0, self.flowdiry)

        # this is to plot the observed flow directions
        # fig, axs = plt.subplots(1, 1, figsize=(8,16))
        # plt.quiver(self.flowdirx,self.flowdiry)
        # axs.axis("equal")

    def optimize(self):
        """
        This function does the data assimilation (inverse modelling) to optimize thk, strflowctrl ans usurf from data
        Check at this [page](https://github.com/jouvetg/igm/blob/main/doc/Inverse-modeling.md)
        """

        self.initialize_iceflow()

        ###### PERFORM CHECKS PRIOR OPTIMIZATIONS

        # make sure this condition is satisfied
        assert ("usurf" in self.config.opti_cost) == (
            "usurf" in self.config.opti_control
        )

        # make sure the loaded ice flow emulator has these inputs
        assert (
            self.iceflow_mapping["fieldin"]
            == ["thk", "slopsurfx", "slopsurfy", "arrhenius", "slidingco"]
        ) | (
            self.iceflow_mapping["fieldin"]
            == ["thk", "slopsurfx", "slopsurfy", "strflowctrl"]
        )

        # make sure the loaded ice flow emulator has at least these outputs
        assert all(
            [
                (f in self.iceflow_mapping["fieldout"])
                for f in ["ubar", "vbar", "uvelsurf", "vvelsurf"]
            ]
        )

        # make sure that there are lease some profiles in thkobs
        if "thk" in self.config.opti_cost:
            if self.config.thk_profiles_file == "":
                assert not tf.reduce_all(tf.math.is_nan(self.thkobs))

        ###### PREPARE DATA PRIOR OPTIMIZATIONS

        if "thk" in self.config.opti_cost:
            if not self.config.thk_profiles_file == "":
                self.load_thk_profiles()

        if hasattr(self, "uvelsurfobs") & hasattr(self, "vvelsurfobs"):
            self.velsurfobs = tf.stack([self.uvelsurfobs, self.vvelsurfobs], axis=-1)

        if "divfluxobs" in self.config.opti_cost:
            self.divfluxobs = self.smb - self.dhdt

        if not self.config.opti_smooth_anisotropy_factor == 1:
            self.compute_flow_direction_for_anisotropic_smoothing()

        if self.config.opti_make_holes_in_data > 0:
            self.make_data_holes()

        if hasattr(self, "thkinit"):
            self.thk.assign(self.thkinit)
        else:
            self.thk.assign(tf.zeros_like(self.thk))

        if self.config.opti_init_zero_thk:
            self.thk.assign(tf.zeros_like(self.thk))

        ###### PREPARE OPIMIZER
        
        if int(tf.__version__.split('.')[1])<=10:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.opti_step_size)
        else:
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.config.opti_step_size)

        # initial_learning_rate * decay_rate ^ (step / decay_steps)
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( initial_learning_rate=opti_step_size, decay_steps=100, decay_rate=0.9)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # add scalng for usurf
        self.iceflow_fieldbounds["usurf"] = (
            self.iceflow_fieldbounds["slopsurfx"] * self.dx
        )

        ###### PREPARE VARIABLES TO OPTIMIZE

        if self.iceflow_mapping["fieldin"] == [
            "thk",
            "slopsurfx",
            "slopsurfy",
            "arrhenius",
            "slidingco",
        ]:
            self.iceflow_fieldbounds["strflowctrl"] = (
                self.iceflow_fieldbounds["arrhenius"]
                + self.iceflow_fieldbounds["slidingco"]
            )

        thk = tf.Variable(self.thk / self.iceflow_fieldbounds["thk"])  # normalized vars
        strflowctrl = tf.Variable(
            self.strflowctrl / self.iceflow_fieldbounds["strflowctrl"]
        )  # normalized vars
        usurf = tf.Variable(
            self.usurf / self.iceflow_fieldbounds["usurf"]
        )  # normalized vars

        self.costs = []

        self.tcomp["Optimize"] = []

        # main loop
        for i in range(self.config.opti_nbitmax):

            with tf.GradientTape() as t:

                self.tcomp["Optimize"].append(time.time())

                # is necessary to remember all operation to derive the gradients w.r.t. control variables
                if "thk" in self.config.opti_control:
                    t.watch(thk)
                if "usurf" in self.config.opti_control:
                    t.watch(usurf)
                if "strflowctrl" in self.config.opti_control:
                    t.watch(strflowctrl)

                # update surface gradient
                if (i == 0) | ("usurf" in self.config.opti_control):
                    slopsurfx, slopsurfy = self.compute_gradient_tf(
                        usurf * self.iceflow_fieldbounds["usurf"], self.dx, self.dx
                    )
                    slopsurfx = slopsurfx / self.iceflow_fieldbounds["slopsurfx"]
                    slopsurfy = slopsurfy / self.iceflow_fieldbounds["slopsurfy"]

                if self.iceflow_mapping["fieldin"] == [
                    "thk",
                    "slopsurfx",
                    "slopsurfy",
                    "arrhenius",
                    "slidingco",
                ]:

                    thrv = (
                        self.config.opti_thr_strflowctrl
                        / self.iceflow_fieldbounds["strflowctrl"]
                    )
                    arrhenius = tf.where(strflowctrl <= thrv, strflowctrl, thrv)
                    slidingco = tf.where(strflowctrl <= thrv, 0, strflowctrl - thrv)

                    # build input of the emulator
                    X = tf.concat(
                        [
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.pad(thk, self.PAD, "CONSTANT"), axis=0
                                ),
                                axis=-1,
                            ),
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.pad(slopsurfx, self.PAD, "CONSTANT"), axis=0
                                ),
                                axis=-1,
                            ),
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.pad(slopsurfy, self.PAD, "CONSTANT"), axis=0
                                ),
                                axis=-1,
                            ),
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.pad(arrhenius, self.PAD, "CONSTANT"), axis=0
                                ),
                                axis=-1,
                            ),
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.pad(slidingco, self.PAD, "CONSTANT"), axis=0
                                ),
                                axis=-1,
                            ),
                        ],
                        axis=-1,
                    )

                elif self.iceflow_mapping["fieldin"] == [
                    "thk",
                    "slopsurfx",
                    "slopsurfy",
                    "strflowctrl",
                ]:

                    # build input of the emulator
                    X = tf.concat(
                        [
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.pad(thk, self.PAD, "CONSTANT"), axis=0
                                ),
                                axis=-1,
                            ),
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.pad(slopsurfx, self.PAD, "CONSTANT"), axis=0
                                ),
                                axis=-1,
                            ),
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.pad(slopsurfy, self.PAD, "CONSTANT"), axis=0
                                ),
                                axis=-1,
                            ),
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.pad(strflowctrl, self.PAD, "CONSTANT"), axis=0
                                ),
                                axis=-1,
                            ),
                        ],
                        axis=-1,
                    )
                else:
                    # ONLY these 2 above cases were implemented !!!
                    sys.exit(
                        "CHANGE THE ICE FLOW EMULATOR -- IMCOMPATIBLE FOR INVERSION "
                    )

                # evalutae th ice flow emulator
                Y = self.iceflow_model(X)

                # get the dimensions of the working array
                Ny, Nx = self.thk.shape

                # save output variables into self.variables for outputs
                for kk, f in enumerate(self.iceflow_mapping["fieldout"]):
                    vars(self)[f].assign(
                        Y[0, :Ny, :Nx, kk] * self.iceflow_fieldbounds[f]
                    )

                # find index of variables in output
                iubar = self.iceflow_mapping["fieldout"].index("ubar")
                ivbar = self.iceflow_mapping["fieldout"].index("vbar")
                iuvsu = self.iceflow_mapping["fieldout"].index("uvelsurf")
                ivvsu = self.iceflow_mapping["fieldout"].index("vvelsurf")

                # save output of the emaultor to compute the costs function
                ubar = (
                    Y[0, :Ny, :Nx, iubar] * self.iceflow_fieldbounds["ubar"]
                )  # NOT normalized vars
                vbar = (
                    Y[0, :Ny, :Nx, ivbar] * self.iceflow_fieldbounds["vbar"]
                )  # NOT normalized vars
                uvelsurf = (
                    Y[0, :Ny, :Nx, iuvsu] * self.iceflow_fieldbounds["uvelsurf"]
                )  # NOT normalized vars
                vvelsurf = (
                    Y[0, :Ny, :Nx, ivvsu] * self.iceflow_fieldbounds["vvelsurf"]
                )  # NOT normalized vars
                velsurf = tf.stack([uvelsurf, vvelsurf], axis=-1)  # NOT normalized vars

                # misfit between surface velocity
                if "velsurf" in self.config.opti_cost:
                    ACT = ~tf.math.is_nan(self.velsurfobs)
                    COST_U = 0.5 * tf.reduce_mean(
                        (
                            (self.velsurfobs[ACT] - velsurf[ACT])
                            / self.config.opti_velsurfobs_std
                        )
                        ** 2
                    )
                else:
                    COST_U = tf.Variable(0.0)

                # misfit between ice thickness profiles
                if "thk" in self.config.opti_cost:
                    if self.config.thk_profiles_file == "":
                        ACT = ~tf.math.is_nan(self.thkobs)
                        COST_H = 0.5 * tf.reduce_mean(
                            (
                                (
                                    self.thkobs[ACT]
                                    - thk[ACT] * self.iceflow_fieldbounds["thk"]
                                )
                                / self.config.opti_thkobs_std
                            )
                            ** 2
                        )
                    else:
                        import tensorflow_addons as tfa

                        COST_H = 0.5 * tf.reduce_mean(
                            (
                                (
                                    self.thktargetprof
                                    - tfa.image.interpolate_bilinear(
                                        tf.expand_dims(
                                            tf.expand_dims(
                                                thk * self.iceflow_fieldbounds["thk"],
                                                axis=0,
                                            ),
                                            axis=-1,
                                        ),
                                        self.jiptprof,
                                        indexing="ij",
                                    )[0, :, 0]
                                )
                                / self.thktargetprof_std
                            )
                            ** 2
                        )
                else:
                    COST_H = tf.Variable(0.0)

                # misfit divergence of the flux
                if ("divfluxobs" in self.config.opti_cost) | (
                    "divfluxfcz" in self.config.opti_cost
                ):

                    divflux = self.compute_divflux(
                        ubar,
                        vbar,
                        thk * self.iceflow_fieldbounds["thk"],
                        self.dx,
                        self.dx,
                    )

                    if "divfluxfcz" in self.config.opti_cost:
                        ACT = self.icemaskobs > 0.5
                        if i % 10 == 0:
                            # his does not need to be comptued any iteration as this is expensive
                            res = stats.linregress(
                                self.usurf[ACT], divflux[ACT]
                            )  # this is a linear regression (usually that's enough)
                        # or you may go for polynomial fit (more gl, but may leads to errors)
                        #  weights = np.polyfit(self.usurf[ACT],divflux[ACT], 2)
                        divfluxtar = tf.where(
                            ACT, res.intercept + res.slope * self.usurf, 0.0
                        )
                    #                        divfluxtar = tf.where(ACT, np.poly1d(weights)(self.usurf) , 0.0 )

                    else:
                        divfluxtar = self.divfluxobs

                    ACT = self.icemaskobs > 0.5
                    COST_D = 0.5 * tf.reduce_mean(
                        (
                            (divfluxtar[ACT] - divflux[ACT])
                            / self.config.opti_divfluxobs_std
                        )
                        ** 2
                    )

                else:
                    COST_D = tf.Variable(0.0)

                # misfit between top ice surfaces
                if "usurf" in self.config.opti_cost:
                    ACT = self.icemaskobs > 0.5
                    COST_S = 0.5 * tf.reduce_mean(
                        (
                            (
                                usurf[ACT] * self.iceflow_fieldbounds["usurf"]
                                - self.usurfobs[ACT]
                            )
                            / self.config.opti_usurfobs_std
                        )
                        ** 2
                    )
                else:
                    COST_S = tf.Variable(0.0)

                # force usurf = usurf - topg
                if "topg" in self.config.opti_cost:
                    ACT = self.icemaskobs == 1
                    COST_T = 10 ** 10 * tf.reduce_mean(
                        (
                            usurf[ACT] * self.iceflow_fieldbounds["usurf"]
                            - thk[ACT] * self.iceflow_fieldbounds["thk"]
                            - self.topg[ACT]
                        )
                        ** 2
                    )
                else:
                    COST_T = tf.Variable(0.0)

                # force zero thikness outisde the mask
                if "icemask" in self.config.opti_cost:
                    COST_O = 10 ** 10 * tf.math.reduce_mean(
                        tf.where(self.icemaskobs > 0.5, 0.0, thk ** 2)
                    )
                else:
                    COST_O = tf.Variable(0.0)

                # Here one enforces non-negative ice thickness, and possibly zero-thickness in user-defined ice-free areas.
                if "thk" in self.config.opti_control:
                    COST_HPO = 10 ** 10 * tf.math.reduce_mean(
                        tf.where(thk >= 0, 0.0, thk ** 2)
                    )
                else:
                    COST_HPO = tf.Variable(0.0)

                # # Make sur to keep reasonable values for strflowctrl
                if "strflowctrl" in self.config.opti_control:
                    COST_STR = 0.5 * tf.reduce_mean(
                        (
                            (
                                strflowctrl * self.iceflow_fieldbounds["strflowctrl"]
                                - self.config.opti_thr_strflowctrl
                            )
                            / self.config.opti_strflowctrl_std
                        )
                        ** 2
                    )
                else:
                    COST_STR = tf.Variable(0.0)

                # Here one adds a regularization terms for the ice thickness to the cost function
                if "thk" in self.config.opti_control:
                    if self.config.opti_smooth_anisotropy_factor == 1:
                        dbdx = thk[:, 1:] - thk[:, :-1]
                        dbdy = thk[1:, :] - thk[:-1, :]
                        REGU_H = self.config.opti_regu_param_thk * (
                            tf.nn.l2_loss(dbdx) + tf.nn.l2_loss(dbdy)
                        )
                    else:
                        dbdx = thk[:, 1:] - thk[:, :-1]
                        dbdx = (dbdx[1:, :] + dbdx[:-1, :]) / 2.0
                        dbdy = thk[1:, :] - thk[:-1, :]
                        dbdy = (dbdy[:, 1:] + dbdy[:, :-1]) / 2.0
                        REGU_H = self.config.opti_regu_param_thk * (
                            tf.nn.l2_loss((dbdx * self.flowdirx + dbdy * self.flowdiry))
                            + self.config.opti_smooth_anisotropy_factor
                            * tf.nn.l2_loss(
                                (dbdx * self.flowdiry - dbdy * self.flowdirx)
                            )
                            - self.config.opti_convexity_weight
                            * tf.math.reduce_sum(thk)
                        )
                else:
                    REGU_H = tf.Variable(0.0)

                # Here one adds a regularization terms for strflowctrl to the cost function
                if "strflowctrl" in self.config.opti_control:
                    dadx = tf.math.abs(strflowctrl[:, 1:] - strflowctrl[:, :-1])
                    dady = tf.math.abs(strflowctrl[1:, :] - strflowctrl[:-1, :])
                    dadx = tf.where(
                        (self.icemaskobs[:, 1:] > 0.5)
                        & (self.icemaskobs[:, :-1] > 0.5),
                        dadx,
                        0.0,
                    )
                    dady = tf.where(
                        (self.icemaskobs[1:, :] > 0.5)
                        & (self.icemaskobs[:-1, :] > 0.5),
                        dady,
                        0.0,
                    )
                    REGU_A = self.config.opti_regu_param_strflowctrl * (
                        tf.nn.l2_loss(dadx) + tf.nn.l2_loss(dady)
                    )
                else:
                    REGU_A = tf.Variable(0.0)

                # sum all component into the main cost function
                COST = (
                    COST_U
                    + COST_H
                    + COST_D
                    + COST_S
                    + COST_T
                    + COST_O
                    + COST_HPO
                    + COST_STR
                    + REGU_H
                    + REGU_A
                )

                vol = (
                    np.sum(thk * self.iceflow_fieldbounds["thk"])
                    * (self.dx ** 2)
                    / 10 ** 9
                )

                if i % self.config.opti_output_freq == 0:
                    print(
                        " OPTI, step %5.0f , ICE_VOL: %7.2f , COST_U: %7.2f , COST_H: %7.2f , COST_D : %7.2f , COST_S : %7.2f , REGU_H : %7.2f , REGU_A : %7.2f "
                        % (
                            i,
                            vol,
                            COST_U.numpy(),
                            COST_H.numpy(),
                            COST_D.numpy(),
                            COST_S.numpy(),
                            REGU_H.numpy(),
                            REGU_A.numpy(),
                        )
                    )

                self.costs.append(
                    [
                        COST_U.numpy(),
                        COST_H.numpy(),
                        COST_D.numpy(),
                        COST_S.numpy(),
                        REGU_H.numpy(),
                        REGU_A.numpy(),
                    ]
                )

                var_to_opti = []
                if "thk" in self.config.opti_control:
                    var_to_opti.append(thk)
                if "usurf" in self.config.opti_control:
                    var_to_opti.append(usurf)
                if "strflowctrl" in self.config.opti_control:
                    var_to_opti.append(strflowctrl)

                # Compute gradient of COST w.r.t. X
                grads = tf.Variable(t.gradient(COST, var_to_opti))

                # this serve to restict the optimization of controls to the mask
                for ii in range(grads.shape[0]):
                    grads[ii].assign(tf.where((self.icemaskobs > 0.5), grads[ii], 0))

                # One step of descent -> this will update input variable X
                optimizer.apply_gradients(
                    zip([grads[i] for i in range(grads.shape[0])], var_to_opti)
                )

                # get back optimized variables in the pool of self.variables
                if "thk" in self.config.opti_control:
                    self.thk.assign(thk * self.iceflow_fieldbounds["thk"])
                    self.thk.assign(tf.where(self.thk < 0.01, 0, self.thk))
                if "strflowctrl" in self.config.opti_control:
                    self.strflowctrl.assign(
                        strflowctrl * self.iceflow_fieldbounds["strflowctrl"]
                    )
                if "usurf" in self.config.opti_control:
                    self.usurf.assign(usurf * self.iceflow_fieldbounds["usurf"])

                self.divflux = self.compute_divflux(
                    self.ubar, self.vbar, self.thk, self.dx, self.dx
                )

                self.compute_rms_std_optimization(i)

                self.tcomp["Optimize"][-1] -= time.time()
                self.tcomp["Optimize"][-1] *= -1

                if i % self.config.opti_output_freq == 0:
                    self.update_plot_inversion(i, self.config.plot_live)
                    self.update_ncdf_optimize(i)
                # self.update_plot_profiles(i,plot_live)

                # stopping criterion: stop if the cost no longer decrease
                # if i>self.config.opti_nbitmin:
                #     cost = [c[0] for c in costs]
                #     if np.mean(cost[-10:])>np.mean(cost[-20:-10]):
                #         break;

        # now that the ice thickness is optimized, we can fix the bed once for all!
        self.topg.assign(self.usurf - self.thk)

        self.output_ncdf_optimize_final()

        self.plot_cost_functions(self.costs, self.config.plot_live)

        np.savetxt(
            os.path.join(self.config.working_dir, "costs.dat"),
            np.stack(self.costs),
            fmt="%.10f",
            header="        COST_U        COST_H      COST_D       COST_S       REGU_H       REGU_A          HPO ",
        )

        np.savetxt(
            os.path.join(self.config.working_dir, "rms_std.dat"),
            np.stack(
                [
                    self.rmsthk,
                    self.stdthk,
                    self.rmsvel,
                    self.stdvel,
                    self.rmsdiv,
                    self.stddiv,
                    self.rmsusurf,
                    self.stdusurf,
                ],
                axis=-1,
            ),
            fmt="%.10f",
            header="        rmsthk      stdthk       rmsvel       stdvel       rmsdiv       stddiv       rmsusurf       stdusurf",
        )

        np.savetxt(
            os.path.join(self.config.working_dir, "strflowctrl.dat"),
            np.array(
                [
                    np.mean(self.strflowctrl[self.icemaskobs > 0.5]),
                    np.std(self.strflowctrl[self.icemaskobs > 0.5]),
                ]
            ),
            fmt="%.3f",
        )

        np.savetxt(
            os.path.join(self.config.working_dir, "volume.dat"),
            np.array([np.sum(self.thk) * self.dx * self.dx / (10 ** 9)]),
            fmt="%.3f",
        )

        np.savetxt(
            os.path.join(self.config.working_dir, "tcompoptimize.dat"),
            np.array([np.sum([f for f in self.tcomp["Optimize"]])]),
            fmt="%.3f",
        )

        os.system('echo rm '+ os.path.join(self.config.working_dir, "strflowctrl.dat") + ' >> clean.sh')        
        os.system('echo rm '+ os.path.join(self.config.working_dir, "rms_std.dat") + ' >> clean.sh')        
        os.system('echo rm '+ os.path.join(self.config.working_dir, "costs.dat") + ' >> clean.sh')
        os.system('echo rm '+ os.path.join(self.config.working_dir, "volume.dat") + ' >> clean.sh')
        os.system('echo rm '+ os.path.join(self.config.working_dir, "tcompoptimize.dat") + ' >> clean.sh')

    def run_opti(self):

        self.initialize()

        with tf.device(self.device_name):

            self.load_ncdf_data(self.config.observation_file)

            self.initialize_fields()

            self.optimize()

    def compute_rms_std_optimization(self, i):
        """
        compute_std_optimization
        """

        I = self.icemaskobs == 1

        if i == 0:
            self.rmsthk = []
            self.stdthk = []
            self.rmsvel = []
            self.stdvel = []
            self.rmsusurf = []
            self.stdusurf = []
            self.rmsdiv = []
            self.stddiv = []

        if hasattr(self, "profile") | hasattr(self, "thkobs"):
            if self.config.thk_profiles_file == "":
                ACT = ~tf.math.is_nan(self.thkobs)
                if np.sum(ACT) == 0:
                    self.rmsthk.append(0)
                    self.stdthk.append(0)
                else:
                    self.rmsthk.append(np.nanmean(self.thk[ACT] - self.thkobs[ACT]))
                    self.stdthk.append(np.nanstd(self.thk[ACT] - self.thkobs[ACT]))

            else:
                fthk = RectBivariateSpline(self.x, self.y, np.transpose(self.thk))
                thkdiff = np.concatenate(
                    [
                        (fthk(p[:, 1], p[:, 2], grid=False) - p[:, 3])
                        for p in self.profile
                    ]
                )
                self.rmsthk.append(np.mean(thkdiff))
                self.stdthk.append(np.std(thkdiff))
        else:
            self.rmsthk.append(0)
            self.stdthk.append(0)

        if hasattr(self, "uvelsurfobs"):
            velsurf_mag = self.getmag(self.uvelsurf, self.vvelsurf).numpy()
            velsurfobs_mag = self.getmag(self.uvelsurfobs, self.vvelsurfobs).numpy()
            ACT = ~np.isnan(velsurfobs_mag)

            self.rmsvel.append(
                np.mean(
                    velsurf_mag[(I & ACT).numpy()] - velsurfobs_mag[(I & ACT).numpy()]
                )
            )
            self.stdvel.append(
                np.std(
                    velsurf_mag[(I & ACT).numpy()] - velsurfobs_mag[(I & ACT).numpy()]
                )
            )
        else:
            self.rmsvel.append(0)
            self.stdvel.append(0)

        if hasattr(self, "divfluxobs"):
            self.rmsdiv.append(np.mean(self.divfluxobs[I] - self.divflux[I]))
            self.stddiv.append(np.std(self.divfluxobs[I] - self.divflux[I]))
        else:
            self.rmsdiv.append(0)
            self.stddiv.append(0)

        if hasattr(self, "usurfobs"):
            self.rmsusurf.append(np.mean(self.usurf[I] - self.usurfobs[I]))
            self.stdusurf.append(np.std(self.usurf[I] - self.usurfobs[I]))
        else:
            self.rmsusurf.append(0)
            self.stdusurf.append(0)

    def update_ncdf_optimize(self, it):
        """
        Initialize and write the ncdf optimze file
        """

        if self.config.verbosity == 1:
            print("Initialize  and write NCDF output Files")

        if "arrhenius" in self.config.opti_vars_to_save:
            self.arrhenius = tf.where(
                self.strflowctrl <= self.config.opti_thr_strflowctrl,
                self.strflowctrl,
                self.config.opti_thr_strflowctrl,
            )

        if "slidingco" in self.config.opti_vars_to_save:
            self.slidingco = tf.where(
                self.strflowctrl <= self.config.opti_thr_strflowctrl,
                0,
                self.strflowctrl - self.config.opti_thr_strflowctrl,
            )

        if "topg" in self.config.opti_vars_to_save:
            self.topg.assign(self.usurf - self.thk)

        if "velsurf_mag" in self.config.opti_vars_to_save:
            self.velsurf_mag = self.getmag(self.uvelsurf, self.vvelsurf)

        if "velsurfobs_mag" in self.config.opti_vars_to_save:
            self.velsurfobs_mag = self.getmag(self.uvelsurfobs, self.vvelsurfobs)

        if it == 0:

            nc = Dataset(
                os.path.join(self.config.working_dir, "optimize.nc"),
                "w",
                format="NETCDF4",
            )

            nc.createDimension("iterations", None)
            E = nc.createVariable("iterations", np.dtype("float32").char, ("iterations",))
            E.units = "None"
            E.long_name = "iterations"
            E.axis = "ITERATIONS"
            E[0] = it

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

            for var in self.config.opti_vars_to_save:

                E = nc.createVariable(
                    var, np.dtype("float32").char, ("iterations", "y", "x")
                )
                E[0, :, :] = vars(self)[var].numpy()

            nc.close()
            
            os.system('echo rm '+ os.path.join(self.config.working_dir, "optimize.nc") + ' >> clean.sh')

        else:

            nc = Dataset(
                os.path.join(self.config.working_dir, "optimize.nc"),
                "a",
                format="NETCDF4",
            )

            d = nc.variables["iterations"][:].shape[0]

            nc.variables["iterations"][d] = it

            for var in self.config.opti_vars_to_save:
                nc.variables[var][d, :, :] = vars(self)[var].numpy()

            nc.close()

    def output_ncdf_optimize_final(self):
        """
        Write final geology after optimizing
        """

        if self.config.verbosity == 1:
            print("Write the final geology ncdf file after optimization")

        nc = Dataset(
            os.path.join(self.config.working_dir, self.config.observation_file), "r"
        )
        varori = [v for v in nc.variables]
        nc.close()

        varori.remove("x")
        varori.remove("y")
        if not "strflowctrl" in varori:
            varori.append("strflowctrl")
        if not "arrhenius" in varori:
            varori.append("arrhenius")
        if not "slidingco" in varori:
            varori.append("slidingco")
        if not "thk" in varori:
            varori.append("thk")
        if not "usurf" in varori:
            varori.append("usurf")
        if not "icemask" in varori:
            varori.append("icemask")

        self.arrhenius = tf.where(
            self.strflowctrl <= self.config.opti_thr_strflowctrl,
            self.strflowctrl,
            self.config.opti_thr_strflowctrl,
        )
        self.slidingco = tf.where(
            self.strflowctrl <= self.config.opti_thr_strflowctrl,
            0,
            self.strflowctrl - self.config.opti_thr_strflowctrl,
        )
        self.velsurf_mag = self.getmag(self.uvelsurf, self.vvelsurf)

        self.icemask = tf.where(
            self.thk > 1.0, tf.ones_like(self.thk), tf.zeros_like(self.thk)
        )

        nc = Dataset(
            os.path.join(self.config.working_dir, self.config.geology_optimized_file),
            "w",
            format="NETCDF4",
        )

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

        for var in varori:

            if hasattr(self, var):
                E = nc.createVariable(var, np.dtype("float32").char, ("y", "x"))
                #                E.long_name = self.var_info[var][0]
                #                E.units     = self.var_info[var][1]
                E[:, :] = vars(self)[var].numpy()

        nc.close()
        
        os.system('echo rm '+ os.path.join(self.config.working_dir, self.config.geology_optimized_file) + ' >> clean.sh')

    def plot_cost_functions(self, costs, plot_live):

        costs = np.stack(costs)

        for i in range(costs.shape[1]):
            costs[:, i] -= np.min(costs[:, i])
            costs[:, i] /= np.max(costs[:, i])

        fig = plt.figure(figsize=(10, 10))
        plt.plot(costs[:, 0], "-k", label="COST U")
        plt.plot(costs[:, 1], "-r", label="COST H")
        plt.plot(costs[:, 2], "-b", label="COST D")
        plt.plot(costs[:, 3], "-g", label="COST S")
        plt.plot(costs[:, 4], "--c", label="REGU H")
        plt.plot(costs[:, 5], "--m", label="REGU A")
        plt.ylim(0, 1)
        plt.legend()

        if plot_live:
            plt.show()
        else:
            plt.savefig(
                os.path.join(self.config.working_dir, "convergence.png"), pad_inches=0
            )
            plt.close("all")
            
            os.system('echo rm '+ os.path.join(self.config.working_dir, "convergence.png") + ' >> clean.sh')

    def update_plot_inversion(self, i, plot_live):
        """
        Plot thickness, velocity, mand slidingco
        """

        if self.config.plot_result:

            if hasattr(self, "uvelsurfobs"):
                velsurfobs_mag = self.getmag(self.uvelsurfobs, self.vvelsurfobs).numpy()
            else:
                velsurfobs_mag = np.zeros_like(self.thk.numpy())

            if hasattr(self, "usurfobs"):
                usurfobs = self.usurfobs
            else:
                usurfobs = np.zeros_like(self.thk.numpy())

            ########################################################

            fig = plt.figure(figsize=(18, 13))

            #########################################################

            ax = fig.add_subplot(2, 3, 1)
            extent = [self.x[0], self.x[-1], self.y[0], self.y[-1]]
            im1 = ax.imshow(self.thk, origin="lower", extent=extent, vmin=0, vmax=800)
            plt.colorbar(im1)

            if hasattr(self, "profile"):
                fthk = RectBivariateSpline(self.x, self.y, np.transpose(self.thk))
                for j, p in enumerate(self.profile):
                    if j > 0:
                        meanfitprofile = np.mean(
                            fthk(p[:, 1], p[:, 2], grid=False) - p[:, 3]
                        )
                        ax.scatter(p[:, 1], p[:, 2], c="k", s=1)
                        ax.text(
                            np.mean(p[:, 1]),
                            np.mean(p[:, 2]),
                            str(int(meanfitprofile)),
                            fontsize=15,
                        )

            ax.set_title(
                "THK, RMS : "
                + str(int(self.rmsthk[-1]))
                + ", STD : "
                + str(int(self.stdthk[-1])),
                size=15,
            )
            ax.axis("off")

            #########################################################

            ax = fig.add_subplot(2, 3, 2)
            velsurf_mag = self.getmag(self.uvelsurf, self.vvelsurf).numpy()
            im1 = ax.imshow(
                velsurf_mag, origin="lower", vmin=0, vmax=np.nanmax(velsurfobs_mag)
            )
            plt.colorbar(im1, format="%.2f")
            ax.set_title(
                "MOD VEL, RMS : "
                + str(int(self.rmsvel[-1]))
                + ", STD : "
                + str(int(self.stdvel[-1])),
                size=15,
            )
            ax.axis("off")

            ########################################################

            ax = fig.add_subplot(2, 3, 3)
            im1 = ax.imshow(self.divflux, origin="lower", vmin=-15, vmax=5)
            plt.colorbar(im1, format="%.2f")
            ax.set_title(
                "MOD DIV, RMS : %5.1f , STD : %5.1f"
                % (self.rmsdiv[-1], self.stddiv[-1]),
                size=15,
            )
            ax.axis("off")

            #########################################################

            ax = fig.add_subplot(2, 3, 4)
            im1 = ax.imshow(self.usurf - usurfobs, origin="lower", vmin=-10, vmax=10)
            plt.colorbar(im1, format="%.2f")
            ax.set_title(
                "DELTA USURF, RMS : %5.1f , STD : %5.1f"
                % (self.rmsusurf[-1], self.stdusurf[-1]),
                size=15,
            )
            ax.axis("off")

            ########################################################

            ax = fig.add_subplot(2, 3, 5)
            im1 = ax.imshow(
                velsurfobs_mag, origin="lower", vmin=0, vmax=np.nanmax(velsurfobs_mag)
            )
            plt.colorbar(im1, format="%.2f")
            ax.set_title("OBS VEL (TARGET)", size=15)
            ax.axis("off")

            #######################################################

            ax = fig.add_subplot(2, 3, 6)
            im1 = ax.imshow(self.strflowctrl, origin="lower", vmin=0, vmax=20)
            plt.colorbar(im1, format="%.2f")
            ax.set_title("strflowctrl", size=15)
            ax.axis("off")

            #########################################################

            plt.tight_layout()

            if plot_live:
                plt.show()
            else:
                plt.savefig(
                    os.path.join(
                        self.config.working_dir, "resu-opti-" + str(i).zfill(4) + ".png"
                    ),
                    pad_inches=0,
                )
                plt.close("all")
                
                os.system('echo rm '+ os.path.join(self.config.working_dir, "*.png") + ' >> clean.sh')

    def plot_opti_diff(self, file, vari, plot_live=True):
        """
        Plot thickness, velocity, mand slidingco
        """

        from matplotlib import cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.pyplot as plt
        import xarray as xr

        data = xr.open_dataset(file, engine="netcdf4")[vari]

        minvar = np.min(data[-1])
        minvardiff = minvar / 5
        maxvar = np.max(data[-1])
        maxvardiff = maxvar / 5

        if vari == "divflux":
            maxvar = 10
            maxvardiff = 10
            minvar = -10
            minvardiff = -10

        if vari == "thk":
            maxvardiff = maxvar / 5
            minvardiff = -maxvar / 5

        if vari == "usurf":
            maxvardiff = 10
            minvardiff = -10

        extent = [self.x[0], self.x[-1], self.y[0], self.y[-1]]

        dicc = {}
        dicc["thk"] = "thk"
        dicc["velsurf_mag"] = "vel"
        dicc["usurf"] = "usurf"
        dicc["divflux"] = "div"

        ########################################################

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6), dpi=200)

        if vari in ["velsurf_mag"]:  # ,'thk'
            rms = str(int(vars(self)["rms" + dicc[vari]][0]))
            std = str(int(vars(self)["std" + dicc[vari]][0]))
            comp = " (RMS: " + rms + ", STD: " + std + ")"
        else:
            comp = ""

        ax1.set_title("Initial " + vari + comp)
        im1 = ax1.imshow(
            np.where(self.thk > 1, data[0], np.nan),
            origin="lower",
            extent=extent,
            vmin=minvar,
            vmax=maxvar,
            cmap=cm.get_cmap("viridis", 8),
        )
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = plt.colorbar(im1, format="%.0f", cax=cax1, orientation="vertical")
        ax1.axis("off")

        if vari in ["velsurf_mag", "thk"]:
            rms = str(int(vars(self)["rms" + dicc[vari]][-1]))
            std = str(int(vars(self)["std" + dicc[vari]][-1]))
            comp = " (RMS: " + rms + ", STD: " + std + ")"
        else:
            comp = ""

        ax2.set_title("Optimized " + vari + comp)
        im2 = ax2.imshow(
            np.where(self.thk > 1, data[-1], np.nan),
            origin="lower",
            extent=extent,
            vmin=minvar,
            vmax=maxvar,
            cmap=cm.get_cmap("viridis", 8),
        )
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar2 = plt.colorbar(im2, format="%.0f", cax=cax2, orientation="vertical")
        ax2.axis("off")

        if vari == "velsurf_mag":
            velsurfobs_mag = xr.open_dataset(file, engine="netcdf4")["velsurfobs_mag"]
            tod = np.where(self.thk > 1, velsurfobs_mag[0], np.nan)
            ax3.set_title("Target")
            im3 = ax3.imshow(
                tod,
                origin="lower",
                extent=extent,
                vmin=minvar,
                vmax=maxvar,
                cmap=cm.get_cmap("viridis", 8),
            )
            divider = make_axes_locatable(ax3)
            cax3 = divider.append_axes("right", size="5%", pad=0.05)
            cbar3 = plt.colorbar(im3, format="%.0f", cax=cax3, orientation="vertical")
            ax3.axis("off")
        else:
            tod = np.where(self.thk > 1, data[-1] - data[0], np.nan)
            ax3.set_title("Optimized - Initial")
            im3 = ax3.imshow(
                tod,
                origin="lower",
                vmin=minvardiff,
                vmax=maxvardiff,
                cmap=cm.get_cmap("RdBu", 10),
            )
            divider = make_axes_locatable(ax3)
            cax3 = divider.append_axes("right", size="5%", pad=0.05)
            cbar3 = plt.colorbar(im3, format="%.0f", cax=cax3, orientation="vertical")
            ax3.axis("off")

        #########################################################

        plt.tight_layout()

        if plot_live:
            plt.show()
        else:
            plt.savefig(
                os.path.join(self.config.working_dir, "resu-opti-" + vari + "-.png"),
                pad_inches=0,
            )
            plt.close("all")
            
            os.system('echo rm '+ os.path.join(self.config.working_dir, "*.png") + ' >> clean.sh')

    def update_plot_profiles(self, i, plot_live):

        from scipy.interpolate import RectBivariateSpline

        fthk = RectBivariateSpline(self.x, self.y, np.transpose(self.thk))

        N = len(self.profile)
        N1 = int(np.sqrt(N)) + 1
        N2 = N1
        fig, axs = plt.subplots(N1, N2, figsize=(N1 * 10, N2 * 5))
        #            fig, axs = plt.subplots(N,1,figsize=(10,N*4))
        for j, p in enumerate(self.profile):
            if j > 0:
                jj = j // N1
                ii = j % N1
                axs[ii, jj].set_title(" PROFILE N° : " + str(j))
                axs[ii, jj].plot(p[:, 0], p[:, 3], "-k")
                axs[ii, jj].plot(p[:, 0], fthk(p[:, 1], p[:, 2], grid=False), "-b")
                axs[ii, jj].axis("equal")
        plt.tight_layout()

        if plot_live:
            plt.show()
        else:
            plt.savefig(
                os.path.join(
                    self.config.working_dir, "S1-pro-" + str(i).zfill(4) + ".png"
                ),
                pad_inches=0,
            )
            plt.close("all")
            
            os.system('echo rm '+ os.path.join(self.config.working_dir, "*.png") + ' >> clean.sh')

    ####################################################################################
    ####################################################################################
    ####################################################################################
    #                              PLOT
    ####################################################################################
    ####################################################################################
    ####################################################################################

    def read_config_param_plot(self):

        self.parser.add_argument(
            "--varplot", 
            type=str, 
            default="velbar_mag", 
            help="variable to plot",
        )
        self.parser.add_argument(
            "--varplot_max",
            type=float,
            default=500,
            help="maximum value of the varplot variable used to adjust the scaling of the colorbar",
        )

    def update_plot(self):
        """
        Plot some variables (e.g. thickness, velocity, mass balance) over time
        """

        if (self.saveresult & self.config.plot_result):
            
            from IPython.display import display, clear_output
            
            self.extent = [np.min(self.x),np.max(self.x),np.min(self.y),np.max(self.y)]

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
                    tf.where(self.thk>1,vars(self)[self.config.varplot],np.nan),
                    origin="lower",
                    cmap="viridis",
                    vmin=0,
                    vmax=self.config.varplot_max,
                    extent=self.extent
                ) 
                self.ip = self.ax.scatter(
                    x=[self.x[0]],
                    y=[self.y[0]],
                    c=[1],
                    vmin=0,
                    vmax=1,
                    s=0.5,
                    cmap="RdBu",
                )
                self.ax.set_title("YEAR : " + str(self.t.numpy()), size=15)
                self.cbar = plt.colorbar(im,label=self.config.varplot)
#                self.ax.set_xlim(427500, 430000)
#                self.ax.set_ylim(5142250,5147050)
            else:
                im = self.ax.imshow(
                    tf.where(self.thk>1,vars(self)[self.config.varplot],np.nan),
                    origin="lower",
                    cmap="viridis",
                    vmin=0,
                    vmax=self.config.varplot_max,
                    extent=self.extent
                )
                if hasattr(self,'xpos'):
                    self.ip.set_visible(False)
                    r = 1
                    self.ip = self.ax.scatter(
                        x=self.xpos[::r],
                        y=self.ypos[::r],
                        c=1 - self.rhpos[::r].numpy(),
                        vmin=0,
                        vmax=1,
                        s=0.5,
                        cmap="RdBu",
                    )
                self.ax.set_title("YEAR : " + str(self.t.numpy()), size=15)

            if self.config.plot_live:
                clear_output(wait=True)
                display(self.fig)

            else:
                plt.savefig(
                    os.path.join(
                        self.config.working_dir,
                        self.config.varplot
                        + "-"
                        + str(self.t.numpy()).zfill(4)
                        + ".png",
                    ),
                    bbox_inches="tight",
                    pad_inches=0.2,
                )
                
                os.system('echo rm '+ os.path.join(self.config.working_dir, "*.png") + ' >> clean.sh')

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
        
        os.system('echo rm '+ os.path.join(self.config.working_dir, "PIE-COMPUTATIONAL.png") + ' >> clean.sh')

    def animate_result(self, file, vari, save=False):
        """
        This function can be called at the end of an IGM run, it permits
        to create an video animation in mp4 by readin the ex.nc ncdf file,
        for a given variable 'vari'
        """

        from IPython.display import HTML, display
        import xarray as xr
        from matplotlib import pyplot as plt, animation

        data = xr.open_dataset(file, engine="netcdf4")[vari]

        minvar = np.min(data)
        maxvar = np.max(data)

        if vari == "divflux":
            maxvar = 10
            minvar = -10

        thk = xr.open_dataset(file, engine="netcdf4")["thk"]

        data = xr.where(thk > 1, data, np.nan)

        ratio = self.y.shape[0] / self.x.shape[0]

        fig, ax = plt.subplots(figsize=(6, 6 * ratio), dpi=200)

        # Plot the initial frame.
        cax = data[0, :, :].plot(
            add_colorbar=True, cmap=plt.cm.get_cmap("jet", 20), vmin=minvar, vmax=maxvar
        )
        ax.axis("off")
        ax.axis("equal")

        # dont' show the original frame
        plt.close()

        def animate(i):
            cax.set_array(data[i, :, :].values.flatten())
            if "iterations" in data.coords.variables:
                ax.set_title("It = " + str(data.coords["iterations"].values[i])[:13])
            else:
                ax.set_title("Time = " + str(data.coords["time"].values[i])[:13])

        ani = animation.FuncAnimation(
            fig, animate, frames=data.shape[0], interval=100
        )  # interval in ms between frames

        HTML(ani.to_html5_video())

        # optionally the animation can be saved in avi
        if save:
            ani.save(file.split(".")[0] + "-" + vari + ".mp4")
            
            os.system('echo rm '+ os.path.join(self.config.working_dir, "*.mp4") + ' >> clean.sh')

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
                
        os.system('echo rm '+ os.path.join(self.config.working_dir, "computational-statistics.txt") + ' >> clean.sh')

    ####################################################################################
    ####################################################################################
    ####################################################################################
    #                              RUN
    ####################################################################################
    ####################################################################################
    ####################################################################################

    def run(self):
        """
        This function wraps the full time-iterative scheme for an IGM forward run
        with an optional inverse step.
        """

        # initilize class Igm
        self.initialize()

        # decide what computation device to use (CPU or GPU)
        with tf.device(self.device_name):

            # load geological data (e.g. topography)
            self.load_ncdf_data(self.config.geology_file)
 
            # initialize all 2D field needed in the following
            self.initialize_fields()

            # if a restarting file exist, then load the data from it
            if len(self.config.restartingfile) > 0:
                self.restart(-1)

            # does the inverse modelling/data assimilation if requested
            if self.config.optimize:
                self.optimize()

            # time loop, with all steps
            while self.t.numpy() < self.config.tend:

                self.tcomp["All"].append(time.time())

                self.update_climate() # update climate
 
                self.update_smb() # update surface mass balance

                self.update_iceflow() # update ice flow

                # self.update_particles() # update computation of particles

                self.update_t_dt() # update time, and time step

                self.update_thk() # update ice thickness (mass conservation)

                self.update_topg() # update basal surface topography

                self.update_ncdf_ex() # update ex.nc output file

                self.update_ncdf_ts() # update ts.nc output file

                self.update_plot()  # update output plot

                self.print_info()  # print information on the run

                self.tcomp["All"][-1] -= time.time()
                self.tcomp["All"][-1] *= -1

        self.print_all_comp_info() # Print information related to computational efficiency

####################################################################################
####################################################################################
####################################################################################
#   END   END   END   END   END   END   END   END   END   END   END   END   END
####################################################################################
####################################################################################
####################################################################################
