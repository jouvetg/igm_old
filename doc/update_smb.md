

### <h1 align="center" id="title"> Documentation of update_smb </h1>



This function permits to choose between several surface mass balance models:

* A very simple mass balance model based on a few parameters (ELA, ...), whose parameters are defined in file glacier.config.mb_simple_file. This surface mass balance is provided with IGM.

* Users can build their own mass balance routine, and possibly combine them with a climate routine. E.g. in the aletsch-1880-21000 example, both climate and surface mass balance models were customized to implement i) the computation of daily temperature and precipitation 2D fields ii) an accumulation/melt model (PDD-like) that takes the climate input, and transforms them into effective surface mass balance.

* Surface Mass balance can be given in the form of a neural network, which predicts surface mass balance from temperature and precipitation fields. As an illustration, I have trained a Neural Network from climate and mass balance data from glaciers in the Alps using the [Deep Learning Emulator](https://github.com/jouvetg/dle). To try it, check the example aletsch-1880-2100. Note that this is highly experimental considering that so far i) the training dataset is small ii) no assessment was done.



### <h1 align="center" id="title"> Parameters of update_smb </h1>


``` 

usage: make-doc-function-md.py [-h] [--mb_update_freq MB_UPDATE_FREQ] [--type_mass_balance TYPE_MASS_BALANCE] [--mb_scaling MB_SCALING] [--mb_simple_file MB_SIMPLE_FILE]
                               [--smb_model_lib_path SMB_MODEL_LIB_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --mb_update_freq MB_UPDATE_FREQ
                        Update the mass balance each X years (1)
  --type_mass_balance TYPE_MASS_BALANCE
                        This keywork permits to identify the type of mass balance model, can be: zero, simple, nn or given (Default: simple)
  --mb_scaling MB_SCALING
                        The paramter permit to make a simple mass balance scaling
  --mb_simple_file MB_SIMPLE_FILE
                        Name of the imput file for the simple mass balance model
  --smb_model_lib_path SMB_MODEL_LIB_PATH
                        Model directory in case the smb model in use is 'nn', i.e. neural network
``` 



### <h1 align="center" id="title"> Code of update_smb </h1>


```python 

    def update_smb(self, force=False):
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

