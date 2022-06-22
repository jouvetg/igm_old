

### <h1 align="center" id="title"> Documentation of update_ncdf_ex </h1>



Initialize  and write the ncdf output file



### <h1 align="center" id="title"> Parameters of update_ncdf_ex </h1>


``` 

usage: make-doc-function-md.py [-h] [--vars_to_save VARS_TO_SAVE]

optional arguments:
  -h, --help            show this help message and exit
  --vars_to_save VARS_TO_SAVE
                        List of variables to be recorded in the ncdef file
``` 



### <h1 align="center" id="title"> Code of update_ncdf_ex </h1>


```python 

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

``` 

