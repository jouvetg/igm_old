

### <h1 align="center" id="title"> Documentation of update_thk </h1>


Help on method update_thk in module igm:

update_thk() method of igm.Igm instance
    update ice thickness solving dh/dt + d(u h)/dx + d(v h)/dy = f using
    upwind finite volume, update usurf and slopes



### <h1 align="center" id="title"> Code of update_thk </h1>


```python 

    def update_thk(self):
        """
        update ice thickness solving dh/dt + d(u h)/dx + d(v h)/dy = f using
        upwind finite volume, update usurf and slopes
        """

        if not hasattr(self, "already_called_update_icethickness"):
            self.tcomp["Transport"] = []
            self.already_called_update_icethickness = True

        else:
            if self.config.verbosity == 1:
                print("Ice thickness equation at time : ", self.t.numpy())

            self.tcomp["Transport"].append(time.time())

            # compute the divergence of the flux
            self.divflux = self.compute_divflux(
                self.ubar, self.vbar, self.thk, self.dx, self.dx
            )

            # Forward Euler with projection to keep ice thickness non-negative
            self.thk.assign(
                tf.maximum(self.thk + self.dt * (self.smb - self.divflux), 0)
            )

            self.usurf.assign(self.topg + self.thk)

            self.slopsurfx, self.slopsurfy = self.compute_gradient_tf(
                self.usurf, self.dx, self.dx
            )

            self.tcomp["Transport"][-1] -= time.time()
            self.tcomp["Transport"][-1] *= -1

``` 

