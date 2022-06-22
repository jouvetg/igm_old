

### <h1 align="center" id="title"> Documentation of update_smb </h1>


Help on method update_smb in module igm:

update_smb(force=False) method of igm.Igm instance
    IGM can use several surface mass balance model:
    
    * IGM comes with a very simple mass balance model based on a few parameters (ELA, ...). 
    
    * Users can build their own mass balance routines relatively easily, and possibly combine them with a climate routine (see Climate model below). E.g. in the aletsch-1880-21000 example, both climate and surface mass balance models were customized to implement i) the computation of daily temperature and precipitation 2D fields ii) an accumulation/melt model (PDD-like) that takes the climate input, and transforms them into effective surface mass balance.
    
    * The structure of IGM facilitates the embedding of further emulators beyond the ice flow model assuming that it maps 2D gridded fields to 2D gridded fields similar to the ice flow one. This applies to predicting surface mass balance from temperature and precipitation fields. IGM permits embedding a neural network emulator to model mass balance. As an illustration, I have trained a Convolutional Neural Network (CNN) from climate and mass balance data from glaciers in the Alps using the [Deep Learning Emulator](https://github.com/jouvetg/dle). To try it, check the example aletsch-1880-2100. Note that this is highly experimental considering that so far i) the training dataset is small ii) CNN is overkilled here iii) no assessment was done.



### <h1 align="center" id="title"> Parameters of update_smb </h1>


``` 

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
``` 



### <h1 align="center" id="title"> Code of update_smb </h1>


```python 

    def update_smb(self, force=False):
        """
        IGM can use several surface mass balance model:
        
        * IGM comes with a very simple mass balance model based on a few parameters (ELA, ...). 
        
        * Users can build their own mass balance routines relatively easily, and possibly combine them with a climate routine (see Climate model below). E.g. in the aletsch-1880-21000 example, both climate and surface mass balance models were customized to implement i) the computation of daily temperature and precipitation 2D fields ii) an accumulation/melt model (PDD-like) that takes the climate input, and transforms them into effective surface mass balance.
        
        * The structure of IGM facilitates the embedding of further emulators beyond the ice flow model assuming that it maps 2D gridded fields to 2D gridded fields similar to the ice flow one. This applies to predicting surface mass balance from temperature and precipitation fields. IGM permits embedding a neural network emulator to model mass balance. As an illustration, I have trained a Convolutional Neural Network (CNN) from climate and mass balance data from glaciers in the Alps using the [Deep Learning Emulator](https://github.com/jouvetg/dle). To try it, check the example aletsch-1880-2100. Note that this is highly experimental considering that so far i) the training dataset is small ii) CNN is overkilled here iii) no assessment was done.
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

