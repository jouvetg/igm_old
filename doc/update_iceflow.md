

### <h1 align="center" id="title"> Documentation of update_iceflow </h1>


Help on method update_iceflow in module igm:

update_iceflow() method of igm.Igm instance
    function update the ice flow using the neural network emulator



### <h1 align="center" id="title"> Parameters of update_iceflow </h1>


usage:  [-h] [--iceflow_model_lib_path ICEFLOW_MODEL_LIB_PATH]
        [--multiple_window_size MULTIPLE_WINDOW_SIZE]
        [--force_max_velbar FORCE_MAX_VELBAR]

optional arguments:
  -h, --help            show this help message and exit
  --iceflow_model_lib_path ICEFLOW_MODEL_LIB_PATH
                        model directory
  --multiple_window_size MULTIPLE_WINDOW_SIZE
                        In case the mdel is a unet, it must force window size
                        to be multiple of e.g. 8
  --force_max_velbar FORCE_MAX_VELBAR
                        This permits to artificially upper-bound velocities,
                        active if > 0


### <h1 align="center" id="title"> Code of update_iceflow </h1>


```python 

    def update_iceflow(self):
        """
        function update the ice flow using the neural network emulator
        """

        if self.config.verbosity == 1:
            print("Update ICEFLOW at time : ", self.t)

        if not hasattr(self, "already_called_update_iceflow"):

            self.tcomp["Ice flow"] = []
            self.already_called_update_iceflow = True
            self.initialize_iceflow()

        self.tcomp["Ice flow"].append(time.time())

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

        Y = self.iceflow_model.predict_on_batch(X)

        Ny, Nx = self.thk.shape
        for kk, f in enumerate(self.iceflow_mapping["fieldout"]):
            vars(self)[f].assign(
                tf.where(self.thk > 0, Y[0, :Ny, :Nx, kk], 0)
                * self.iceflow_fieldbounds[f]
            )

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

``` 

