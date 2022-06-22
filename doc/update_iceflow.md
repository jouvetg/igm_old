

### <h1 align="center" id="title"> Documentation of update_iceflow </h1>


Help on method update_iceflow in module igm:

update_iceflow() method of igm.Igm instance
    Ice flow dynamics are modeled using Artificial Neural Networks trained from physical models.
    
    You may find trained and ready-to-use ice flow emulators in the folder `model-lib/T_M_I_Y_V/R/`, where 'T_M_I_Y_V' defines the emulator, and R defines the spatial resolution. Make sure that the resolution of the picked emulator is available in the database. Results produced with IGM will strongly rely on the chosen emulator. Make sure that you use the emulator within the hull of its training dataset (e.g., do not model an ice sheet with an emulator trained with mountain glaciers) to ensure reliability (or fidelity w.r.t to the instructor model) -- the emulator is probably much better at interpolating than at extrapolating. Information on the training dataset is provided in a dedicated README coming along with the emulator.
    
    At the time of writing, I recommend using *f15_cfsflow_GJ_22_a*, which takes ice thickness, top surface slopes, the sliding coefficient c ('slidingco'), and Arrhenuis factor A ('arrhenius'), and return basal, vertical-average and surface x- and y- velocity components as depicted on the following graph: 
    
    ![](https://github.com/jouvetg/igm/blob/main/fig/mapping-f15.png)
    
    I have trained *f17_cfsflow_GJ_22_a* used a large dataset of modeled glaciers (based on a Stokes-based CfsFlow ice flow solver) and varying sliding coefficient c, and Arrhenuis factor A into a 2D space. 
    
    For now, only the emulator trained by CfsFlow and PISM is available with different resolutions. Consider training your own with the [Deep Learning Emulator](https://github.com/jouvetg/dle) if none of these emulators fill your need.



### <h1 align="center" id="title"> Parameters of update_iceflow </h1>


``` 

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
``` 



### <h1 align="center" id="title"> Code of update_iceflow </h1>


```python 

    def update_iceflow(self):
        """
        Ice flow dynamics are modeled using Artificial Neural Networks trained from physical models.
        
        You may find trained and ready-to-use ice flow emulators in the folder `model-lib/T_M_I_Y_V/R/`, where 'T_M_I_Y_V' defines the emulator, and R defines the spatial resolution. Make sure that the resolution of the picked emulator is available in the database. Results produced with IGM will strongly rely on the chosen emulator. Make sure that you use the emulator within the hull of its training dataset (e.g., do not model an ice sheet with an emulator trained with mountain glaciers) to ensure reliability (or fidelity w.r.t to the instructor model) -- the emulator is probably much better at interpolating than at extrapolating. Information on the training dataset is provided in a dedicated README coming along with the emulator.
        
        At the time of writing, I recommend using *f15_cfsflow_GJ_22_a*, which takes ice thickness, top surface slopes, the sliding coefficient c ('slidingco'), and Arrhenuis factor A ('arrhenius'), and return basal, vertical-average and surface x- and y- velocity components as depicted on the following graph: 
        
        ![](https://github.com/jouvetg/igm/blob/main/fig/mapping-f15.png)
        
        I have trained *f17_cfsflow_GJ_22_a* used a large dataset of modeled glaciers (based on a Stokes-based CfsFlow ice flow solver) and varying sliding coefficient c, and Arrhenuis factor A into a 2D space. 
        
        For now, only the emulator trained by CfsFlow and PISM is available with different resolutions. Consider training your own with the [Deep Learning Emulator](https://github.com/jouvetg/dle) if none of these emulators fill your need.
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

