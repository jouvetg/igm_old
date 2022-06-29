

### <h1 align="center" id="title"> Documentation of update_particles </h1>



IGM includes a particle tracking routine, which can compute a large number of trajectories (as it is implemented with TensorFlow to run in parallel) in live time during the forward model run. The routine produces some seeding of particles (by default in the accumulation area at regular intervals), and computes the time trajectory of the resulting particle in time advected by the velocity field in 3D. Horizontal and vertical directions are treated differently:

* In the horizontal plan, particles are advected with the horizontal velocity field (interpolated bi-linearly). The positions are recorded in vector (glacier.xpos,glacier.ypos).

* In the vertical direction, particles are tracked along the ice column scaled between 0 and 1 (0 at the bed, 1 at the top surface). The relative position along the ice column is recorded in vector glacier.rhpos (same dimension as glacier.xpos and iglaciergm.ypos). Particles are always initialized at 1 (assumed to be on the surface). The evolution of the particle within the ice column through time is computed according to the surface mass balance: the particle deepens when the surface mass balance is positive (then igm.rhpos decreases), and re-emerge when the surface mass balance is negative.  

To include this feature, make sure:

* To activate it: glacier.config.tracking_particles=True

* To adapt the seeding to your need. You may keep the default seeding in the accumulation area setting the seeding frequency with igm.config.frequency_seeding and the seeding density glacier.config.density_seeding. Alternatively, you may define your own seeding strategy (e.g. seeding close to rock walls/nunataks). To do so, you may redefine the function seeding_particles.

* At each time step, the weight of surface debris contains in each cell the 2D horizontal grid is computed, and stored in variable igm.weight_particles.

* You may visualize the moving particles in update_plot(). 

* Particles trajectories are saved in folder 'trajectory', and are organized by seeding times.

Check at the example aletsch-1880-2100 for an example of particle tracking.



### <h1 align="center" id="title"> Parameters of update_particles </h1>


``` 

usage: make-doc-function-md.py [-h] [--tracking_particles TRACKING_PARTICLES]
                               [--tracking_method TRACKING_METHOD]
                               [--frequency_seeding FREQUENCY_SEEDING]
                               [--density_seeding DENSITY_SEEDING]

optional arguments:
  -h, --help            show this help message and exit
  --tracking_particles TRACKING_PARTICLES
                        Is the particle tracking active? (Default: False)
  --tracking_method TRACKING_METHOD
                        Method for tracking particles
  --frequency_seeding FREQUENCY_SEEDING
                        Frequency of seeding (default: 10)
  --density_seeding DENSITY_SEEDING
                        Density of seeding (default: 0.2)
``` 



### <h1 align="center" id="title"> Code of update_particles </h1>


```python 

    def update_particles(self):
        """
        IGM includes a particle tracking routine, which can compute a large number of trajectories (as it is implemented with TensorFlow to run in parallel) in live time during the forward model run. The routine produces some seeding of particles (by default in the accumulation area at regular intervals), and computes the time trajectory of the resulting particle in time advected by the velocity field in 3D. Horizontal and vertical directions are treated differently:
        
        * In the horizontal plan, particles are advected with the horizontal velocity field (interpolated bi-linearly). The positions are recorded in vector (glacier.xpos,glacier.ypos).
        
        * In the vertical direction, particles are tracked along the ice column scaled between 0 and 1 (0 at the bed, 1 at the top surface). The relative position along the ice column is recorded in vector glacier.rhpos (same dimension as glacier.xpos and iglaciergm.ypos). Particles are always initialized at 1 (assumed to be on the surface). The evolution of the particle within the ice column through time is computed according to the surface mass balance: the particle deepens when the surface mass balance is positive (then igm.rhpos decreases), and re-emerge when the surface mass balance is negative.  
        
        To include this feature, make sure:
        
        * To activate it: glacier.config.tracking_particles=True
        
        * To adapt the seeding to your need. You may keep the default seeding in the accumulation area setting the seeding frequency with igm.config.frequency_seeding and the seeding density glacier.config.density_seeding. Alternatively, you may define your own seeding strategy (e.g. seeding close to rock walls/nunataks). To do so, you may redefine the function seeding_particles.
        
        * At each time step, the weight of surface debris contains in each cell the 2D horizontal grid is computed, and stored in variable igm.weight_particles.
        
        * You may visualize the moving particles in update_plot(). 
        
        * Particles trajectories are saved in folder 'trajectory', and are organized by seeding times.
        
        Check at the example aletsch-1880-2100 for an example of particle tracking.
        """

        import tensorflow_addons as tfa

        if self.config.verbosity == 1:
            print("Update TRACKING at time : ", self.t)

        if not hasattr(self, "already_called_update_tracking"):
            self.already_called_update_tracking = True
            self.tlast_seeding = -1.0e5000
            self.tcomp["Tracking"] = []

            # initialize trajectories
            self.xpos = tf.Variable([])
            self.ypos = tf.Variable([])
            self.zpos = tf.Variable([])
            self.rhpos = tf.Variable([])
            self.wpos = tf.Variable([])

            # build the gridseed
            self.gridseed = np.zeros_like(self.thk) == 1
            rr = int(1.0 / self.config.density_seeding)
            self.gridseed[::rr, ::rr] = True

            directory = os.path.join(self.config.working_dir, "trajectories")
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.mkdir(directory)

            self.seedtimes = []

        else:

            if (self.t.numpy() - self.tlast_seeding) >= self.config.frequency_seeding:
                self.seeding_particles()

                # merge the new seeding points with the former ones
                self.xpos = tf.Variable(tf.concat([self.xpos, self.nxpos], axis=-1))
                self.ypos = tf.Variable(tf.concat([self.ypos, self.nypos], axis=-1))
                self.zpos = tf.Variable(tf.concat([self.zpos, self.nzpos], axis=-1))
                self.rhpos = tf.Variable(tf.concat([self.rhpos, self.nrhpos], axis=-1))
                self.wpos = tf.Variable(tf.concat([self.wpos, self.nwpos], axis=-1))

                self.tlast_seeding = self.t.numpy()
                self.seedtimes.append([self.t.numpy(), self.xpos.shape[0]])

            self.tcomp["Tracking"].append(time.time())

            # find the indices of trajectories
            # these indicies are real values to permit 2D interpolations
            i = (self.xpos - self.x[0]) / self.dx
            j = (self.ypos - self.y[0]) / self.dx

            indices = tf.expand_dims(
                tf.concat(
                    [tf.expand_dims(j, axis=-1), tf.expand_dims(i, axis=-1)], axis=-1
                ),
                axis=0,
            )

            uvelbase = tfa.image.interpolate_bilinear(
                tf.expand_dims(tf.expand_dims(self.uvelbase, axis=0), axis=-1),
                indices,
                indexing="ij",
            )[0, :, 0]

            vvelbase = tfa.image.interpolate_bilinear(
                tf.expand_dims(tf.expand_dims(self.vvelbase, axis=0), axis=-1),
                indices,
                indexing="ij",
            )[0, :, 0]

            uvelsurf = tfa.image.interpolate_bilinear(
                tf.expand_dims(tf.expand_dims(self.uvelsurf, axis=0), axis=-1),
                indices,
                indexing="ij",
            )[0, :, 0]

            vvelsurf = tfa.image.interpolate_bilinear(
                tf.expand_dims(tf.expand_dims(self.vvelsurf, axis=0), axis=-1),
                indices,
                indexing="ij",
            )[0, :, 0]

            othk = tfa.image.interpolate_bilinear(
                tf.expand_dims(tf.expand_dims(self.thk, axis=0), axis=-1),
                indices,
                indexing="ij",
            )[0, :, 0]
            
            topg = tfa.image.interpolate_bilinear(
                tf.expand_dims(tf.expand_dims(self.topg, axis=0), axis=-1),
                indices,
                indexing="ij",
            )[0, :, 0]
             
            smb = tfa.image.interpolate_bilinear(
                tf.expand_dims(tf.expand_dims(self.smb, axis=0), axis=-1),
                indices,
                indexing="ij",
            )[0, :, 0]
            
            if self.config.tracking_method=='simple':

                nthk = othk + smb * self.dt  # new ice thicnkess after smb update
    
                # adjust the relative height within the ice column with smb
                self.rhpos.assign(
                    tf.where(
                        nthk > 0.1, tf.clip_by_value(self.rhpos * othk / nthk, 0, 1), 1
                    )
                )
                
                uvel = uvelbase + (uvelsurf - uvelbase) * (
                    1 - (1 - self.rhpos) ** 4
                )  # SIA-like
                vvel = vvelbase + (vvelsurf - vvelbase) * (
                    1 - (1 - self.rhpos) ** 4
                )  # SIA-like
    
                self.xpos.assign(self.xpos + self.dt * uvel)  # forward euler
                self.ypos.assign(self.ypos + self.dt * vvel)  # forward euler 
                
                self.zpos.assign( topg + nthk * self.rhpos  )
                
            elif self.config.tracking_method=='3d':
                
                slopsurfx, slopsurfy = self.compute_gradient_tf(self.usurf, self.dx, self.dx)
                sloptopgx, sloptopgy = self.compute_gradient_tf(self.topg, self.dx, self.dx)
        
                self.divflux = self.compute_divflux(self.ubar, self.vbar, self.thk, self.dx, self.dx)
        
                self.wvelbase =                 self.uvelbase * sloptopgx + self.vvelbase * sloptopgy
                self.wvelsurf = -self.divflux + self.uvelsurf * slopsurfx + self.vvelsurf * slopsurfy
                
                wvelbase = tfa.image.interpolate_bilinear(
                    tf.expand_dims(tf.expand_dims(self.wvelbase, axis=0), axis=-1),
                    indices,
                    indexing="ij",
                )[0, :, 0]
                
                wvelsurf = tfa.image.interpolate_bilinear(
                    tf.expand_dims(tf.expand_dims(self.wvelsurf, axis=0), axis=-1),
                    indices,
                    indexing="ij",
                )[0, :, 0]

                self.zpos.assign( tf.clip_by_value( self.zpos , topg, topg+othk) )
                
                self.rhpos.assign( tf.where( othk > 0.1, (self.zpos - topg)/othk , 1 ) )
                 
                uvel = uvelbase + (uvelsurf - uvelbase) * ( 1 - (1 - self.rhpos) ** 4)
                vvel = vvelbase + (vvelsurf - vvelbase) * ( 1 - (1 - self.rhpos) ** 4)
                wvel = wvelbase + (wvelsurf - wvelbase) * ( 1 - (1 - self.rhpos) ** 4)
                   
                self.xpos.assign(self.xpos + self.dt * uvel)  # forward euler
                self.ypos.assign(self.ypos + self.dt * vvel)  # forward euler 
                self.zpos.assign(self.zpos + self.dt * wvel)  # forward euler 
                
            # elif self.config.tracking_method==2:
                 
            #     self.divbase = self.compute_div(self.uvelbase, self.vvelbase, self.dx, self.dx) 
            #     self.divsurf = self.compute_div(self.uvelsurf, self.vvelsurf, self.dx, self.dx) 
                 
            #     divbase = tfa.image.interpolate_bilinear(
            #         tf.expand_dims(tf.expand_dims(self.divbase, axis=0), axis=-1),
            #         indices,
            #         indexing="ij",
            #     )[0, :, 0]
                 
            #     divsurf = tfa.image.interpolate_bilinear(
            #         tf.expand_dims(tf.expand_dims(self.divsurf, axis=0), axis=-1),
            #         indices,
            #         indexing="ij",
            #     )[0, :, 0]
                
            #     divflux = tfa.image.interpolate_bilinear(
            #         tf.expand_dims(tf.expand_dims(self.divflux, axis=0), axis=-1),
            #         indices,
            #         indexing="ij",
            #     )[0, :, 0]
                
            #     topg = tfa.image.interpolate_bilinear(
            #         tf.expand_dims(tf.expand_dims(self.topg, axis=0), axis=-1),
            #         indices,
            #         indexing="ij",
            #     )[0, :, 0]
                
            #     usurf = tfa.image.interpolate_bilinear(
            #         tf.expand_dims(tf.expand_dims(self.usurf, axis=0), axis=-1),
            #         indices,
            #         indexing="ij",
            #     )[0, :, 0]
                
            #     uvel = uvelbase + (uvelsurf - uvelbase) * ( 1 - (1 - self.rhpos) ** 4)
            #     vvel = vvelbase + (vvelsurf - vvelbase) * ( 1 - (1 - self.rhpos) ** 4)
            #     div  = divbase  + (divsurf - divbase) * self.rhpos
                
            #     nthk = tf.maximum( othk + self.dt * (smb - divflux), 0) 
                  
            #     z = (1 - self.rhpos) * othk
                 
            #     self.xpos.assign(self.xpos + self.dt * uvel)  # forward euler
            #     self.ypos.assign(self.ypos + self.dt * vvel)  # forward euler

            #     z  = z + self.dt * ( self.rhpos * smb - z * div )
                
            #     self.rhpos.assign(
            #         tf.clip_by_value( tf.where( nthk > 0.1, 1 - (z/nthk), 1 ) , 0 , 1 )
            #     ) 

            indices = tf.concat(
                [
                    tf.expand_dims(tf.cast(j, dtype="int32"), axis=-1),
                    tf.expand_dims(tf.cast(i, dtype="int32"), axis=-1),
                ],
                axis=-1,
            )
            updates = tf.cast(tf.where(self.rhpos == 1, self.wpos, 0), dtype="float32")
            self.weight_particles = tf.tensor_scatter_nd_add(
                tf.zeros_like(self.thk), indices, updates
            )

            self.update_write_trajectories()

            self.tcomp["Tracking"][-1] -= time.time()
            self.tcomp["Tracking"][-1] *= -1

``` 

