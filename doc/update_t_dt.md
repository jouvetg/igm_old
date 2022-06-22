

### <h1 align="center" id="title"> Documentation of update_t_dt </h1>


Help on method update_t_dt in module igm:

update_t_dt() method of igm.Igm instance
    compute time step to satisfy the CLF condition and hit requested saving times



### <h1 align="center" id="title"> Parameters of update_t_dt </h1>


``` 

usage: [-h] [--cfl CFL] [--dtmax DTMAX]

optional arguments:
  -h, --help     show this help message and exit
  --cfl CFL      CFL number must be below 1 (0.3)
  --dtmax DTMAX  Maximum time step, used only with slow ice (10.0)
``` 



### <h1 align="center" id="title"> Code of update_t_dt </h1>


```python 

    def update_t_dt(self):
        """
        compute time step to satisfy the CLF condition and hit requested saving times
        """
        if self.config.verbosity == 1:
            print("Update DT from the CFL condition at time : ", self.t.numpy())

        if not hasattr(self, "already_called_update_t_dt"):
            self.tcomp["Time step"] = []
            self.already_called_update_t_dt = True

            self.tsave = np.ndarray.tolist(
                np.arange(self.config.tstart, self.config.tend, self.config.tsave)
            ) + [self.config.tend]
            self.itsave = 0

        else:
            self.tcomp["Time step"].append(time.time())

            velomax = max(
                tf.math.reduce_max(tf.math.abs(self.ubar)),
                tf.math.reduce_max(tf.math.abs(self.vbar)),
            ).numpy()

            if velomax > 0:
                self.dt_target = min(
                    self.config.cfl * self.dx / velomax, self.config.dtmax
                )
            else:
                self.dt_target = self.config.dtmax

            self.dt = self.dt_target

            if self.tsave[self.itsave + 1] <= self.t.numpy() + self.dt:
                self.dt = self.tsave[self.itsave + 1] - self.t.numpy()
                self.t.assign(self.tsave[self.itsave + 1])
                self.saveresult = True
                self.itsave += 1
            else:
                self.t.assign(self.t.numpy() + self.dt)
                self.saveresult = False

            self.it += 1

            self.tcomp["Time step"][-1] -= time.time()
            self.tcomp["Time step"][-1] *= -1

``` 

