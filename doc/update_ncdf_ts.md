

### <h1 align="center" id="title"> Documentation of update_ncdf_ts </h1>



This function write time serie variables (ice glaziated area and volume) into the ncdf output file ts.nc



### <h1 align="center" id="title"> Code of update_ncdf_ts </h1>


```python 

    def update_ncdf_ts(self, force=False):
        """
        This function write time serie variables (ice glaziated area and volume) into the ncdf output file ts.nc
        """

        if not hasattr(self, "already_called_update_ncdf_ts"):
            self.already_called_update_ncdf_ts = True

        if force | self.saveresult:

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

``` 

