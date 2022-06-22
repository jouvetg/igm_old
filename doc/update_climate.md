

### <h1 align="center" id="title"> Documentation of update_climate </h1>



Climate forcing can be easily enforced in IGM by customizing this function to your needs, e.g., building fields of temperature and precipitation, which can be used by an accumulation/melt model (PDD-like) model. Check at the aletsch-1880-21000 example.



### <h1 align="center" id="title"> Parameters of update_climate </h1>


``` 

usage: make-doc-function-md.py [-h] [--clim_update_freq CLIM_UPDATE_FREQ]
                               [--type_climate TYPE_CLIMATE]

optional arguments:
  -h, --help            show this help message and exit
  --clim_update_freq CLIM_UPDATE_FREQ
                        Update the climate each X years (1)
  --type_climate TYPE_CLIMATE
                        toy or any custom climate
``` 



### <h1 align="center" id="title"> Code of update_climate </h1>


```python 

    def update_climate(self, force=False):
        """
        Climate forcing can be easily enforced in IGM by customizing this function to your needs, e.g., building fields of temperature and precipitation, which can be used by an accumulation/melt model (PDD-like) model. Check at the aletsch-1880-21000 example.
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

