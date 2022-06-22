

### <h1 align="center" id="title"> Documentation of update_topg </h1>


Help on method update_topg in module igm:

update_topg() method of igm.Igm instance
    update bedrock due to glacial errosion,
    eroson rate are proportional to a power
    of the sliding velocity magnitude



### <h1 align="center" id="title"> Parameters of update_topg </h1>


usage:  [-h] [--erosion_include EROSION_INCLUDE] [--erosion_cst EROSION_CST]
        [--erosion_exp EROSION_EXP]
        [--erosion_update_freq EROSION_UPDATE_FREQ]
        [--uplift_include UPLIFT_INCLUDE] [--uplift_rate UPLIFT_RATE]
        [--uplift_update_freq UPLIFT_UPDATE_FREQ]

optional arguments:
  -h, --help            show this help message and exit
  --erosion_include EROSION_INCLUDE
                        Include a model for bedrock erosion
  --erosion_cst EROSION_CST
                        Herman, F. et al. Erosion by an Alpine glacier.
                        Science 350, 193–195 (2015)
  --erosion_exp EROSION_EXP
                        Herman, F. et al. Erosion by an Alpine glacier.
                        Science 350, 193–195 (2015).
  --erosion_update_freq EROSION_UPDATE_FREQ
                        Update the erosion only each 100 years
  --uplift_include UPLIFT_INCLUDE
                        Include a model for constant bedrock uplift
  --uplift_rate UPLIFT_RATE
                        unit is m/y
  --uplift_update_freq UPLIFT_UPDATE_FREQ
                        Update the uplift only each 100 years


### <h1 align="center" id="title"> Code of update_topg </h1>


```python 

    def update_topg(self):
        """
        update bedrock due to glacial errosion,
        eroson rate are proportional to a power
        of the sliding velocity magnitude
        """

        if not hasattr(self, "already_called_update_topg"):
            self.tlast_erosion = self.config.tstart
            self.tlast_uplift = self.config.tstart
            self.tcomp["Erosion"] = []
            self.tcomp["Uplift"] = []
            self.already_called_update_topg = True

        if self.config.erosion_include:

            if (self.t.numpy() - self.tlast_erosion) >= self.config.erosion_update_freq:

                if self.config.verbosity == 1:
                    print("Erode bedrock at time : ", self.t.numpy())

                self.tcomp["Erosion"].append(time.time())

                self.velbase_mag = self.getmag(self.uvelbase, self.vvelbase)

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

                self.topg.assign(
                    self.topg
                    + self.config.uplift_rate * (self.t.numpy() - self.tlast_uplift)
                )

                self.usurf.assign(self.topg + self.thk)

                self.tlast_uplift = self.t.numpy()

                self.tcomp["Uplift"][-1] -= time.time()
                self.tcomp["Uplift"][-1] *= -1

``` 

