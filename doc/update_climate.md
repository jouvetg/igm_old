

### <h1 align="center" id="title"> Documentation of update_climate </h1>



This function serves to define a climate forcing (e.g. monthly temperature and precipitation fields) to be used for the surface mass balance model (e.g. accumulation/melt PDD-like model). No climate forcing is provided with IGM is this is case-dependent. Check at the aletsch-1880-21000 example.



### <h1 align="center" id="title"> Parameters of update_climate </h1>


``` 

usage: make-doc-function-md.py [-h] [--clim_update_freq CLIM_UPDATE_FREQ] [--type_climate TYPE_CLIMATE]

optional arguments:
  -h, --help            show this help message and exit
  --clim_update_freq CLIM_UPDATE_FREQ
                        Update the climate each X years (default: 1)
  --type_climate TYPE_CLIMATE
                        This keywork serves to identify & call the climate forcing. If an empty string, this function is not called (Default: )
``` 



### <h1 align="center" id="title"> Code of update_climate </h1>


```python 

    def update_climate(self, force=False):
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

            if force | new_clim_needed:

                if self.config.verbosity == 1:
                    print("Construct climate at time : ", self.t)

                self.tcomp["Climate"].append(time.time())

                getattr(self, "update_climate_" + self.config.type_climate)()

                self.tlast_clim = self.t.numpy()

                self.tcomp["Climate"][-1] -= time.time()
                self.tcomp["Climate"][-1] *= -1

``` 

