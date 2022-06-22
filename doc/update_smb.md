

### <h1 align="center" id="title"> Documentation of update_smb </h1>


Help on method update_smb in module igm:

update_smb(force=False) method of igm.Igm instance
    update_mass balance



### <h1 align="center" id="title"> Parameters of update_smb </h1>


usage:  [-h] [--mb_update_freq MB_UPDATE_FREQ]
        [--type_mass_balance TYPE_MASS_BALANCE] [--mb_scaling MB_SCALING]
        [--mb_simple_file MB_SIMPLE_FILE]
        [--smb_model_lib_path SMB_MODEL_LIB_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --mb_update_freq MB_UPDATE_FREQ
                        Update the mass balance each X years (1)
  --type_mass_balance TYPE_MASS_BALANCE
                        zero, simple, given
  --mb_scaling MB_SCALING
                        mass balance scaling
  --mb_simple_file MB_SIMPLE_FILE
                        mb_simple_file
  --smb_model_lib_path SMB_MODEL_LIB_PATH
                        Model directory in case the smb model in use is
                        'nn'for neural netowrk


### <h1 align="center" id="title"> Code of update_smb </h1>


```python 

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

``` 

