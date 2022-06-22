

### <h1 align="center" id="title"> Documentation of update_thk </h1>


Help on method update_thk in module igm:

update_thk() method of igm.Igm instance
    The mass conservation equation is solved using an explicit first-order upwind 
    finite-volume scheme on a regular 2D grid with constant cell spacing in any direction. 
    The discretization and the approximation of the flux divergence is described 
    [here](https://github.com/jouvetg/igm/blob/main/fig/transp-igm.jpg). 
    With this scheme mass of ice is allowed to move from cell to cell (where thickness 
    and velocities are defined) from edge-defined fluxes (inferred from depth-averaged 
    velocities, and ice thickness in upwind direction). 
    The resulting scheme is mass conservative and parallelizable (because fully explicit). 
    However, it is subject to a CFL condition. This means that the time step 
    (defined in glacier.update_t_dt()) is controlled by parameter glacier.config.cfl,
    which is the maximum number of cells crossed in one iteration (this parameter cannot exceed one).



### <h1 align="center" id="title"> Code of update_thk </h1>


```python 

    def update_thk(self):
        """
        The mass conservation equation is solved using an explicit first-order upwind 
        finite-volume scheme on a regular 2D grid with constant cell spacing in any direction. 
        The discretization and the approximation of the flux divergence is described 
        [here](https://github.com/jouvetg/igm/blob/main/fig/transp-igm.jpg). 
        With this scheme mass of ice is allowed to move from cell to cell (where thickness 
        and velocities are defined) from edge-defined fluxes (inferred from depth-averaged 
        velocities, and ice thickness in upwind direction). 
        The resulting scheme is mass conservative and parallelizable (because fully explicit). 
        However, it is subject to a CFL condition. This means that the time step 
        (defined in glacier.update_t_dt()) is controlled by parameter glacier.config.cfl,
        which is the maximum number of cells crossed in one iteration (this parameter cannot exceed one).
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

