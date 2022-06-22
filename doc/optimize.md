

### <h1 align="center" id="title"> Documentation of optimize </h1>



This is the optimization routine to invert thk, strflowctrl ans usurf from data
DEFAULT PARAMETERS ARE
# nbitmin=50, nbitmax=1000, opti_step_size=0.001,
# init_zero_thk=True,
# regu_param_thk=1.0, regu_param_strflowctrl=1.0,
# smooth_anisotropy_factor=0.2, convexity_weight = 0.002,
# opti_control=['thk','strflowctrl'], opti_cost=['velsurf','thk'],



### <h1 align="center" id="title"> Parameters of optimize </h1>


``` 

usage: make-doc-function-md.py [-h] [--opti_vars_to_save OPTI_VARS_TO_SAVE]
                               [--observation_file OBSERVATION_FILE]
                               [--thk_profiles_file THK_PROFILES_FILE] [--mode_opti MODE_OPTI]
                               [--opti_thr_strflowctrl OPTI_THR_STRFLOWCTRL]
                               [--opti_init_zero_thk OPTI_INIT_ZERO_THK]
                               [--opti_regu_param_thk OPTI_REGU_PARAM_THK]
                               [--opti_regu_param_strflowctrl OPTI_REGU_PARAM_STRFLOWCTRL]
                               [--opti_smooth_anisotropy_factor OPTI_SMOOTH_ANISOTROPY_FACTOR]
                               [--opti_convexity_weight OPTI_CONVEXITY_WEIGHT]
                               [--opti_usurfobs_std OPTI_USURFOBS_STD]
                               [--opti_strflowctrl_std OPTI_STRFLOWCTRL_STD]
                               [--opti_velsurfobs_std OPTI_VELSURFOBS_STD]
                               [--opti_thkobs_std OPTI_THKOBS_STD]
                               [--opti_divfluxobs_std OPTI_DIVFLUXOBS_STD]
                               [--opti_control OPTI_CONTROL] [--opti_cost OPTI_COST]
                               [--opti_nbitmin OPTI_NBITMIN] [--opti_nbitmax OPTI_NBITMAX]
                               [--opti_step_size OPTI_STEP_SIZE]
                               [--opti_make_holes_in_data OPTI_MAKE_HOLES_IN_DATA]
                               [--opti_output_freq OPTI_OUTPUT_FREQ]
                               [--geology_optimized_file GEOLOGY_OPTIMIZED_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --opti_vars_to_save OPTI_VARS_TO_SAVE
                        List of variables to be recorded in the ncdef file
  --observation_file OBSERVATION_FILE
                        Observation file contains the 2D data observations fields (thkobs,
                        usurfobs, uvelsurfobs, ....)
  --thk_profiles_file THK_PROFILES_FILE
                        Provide ice thickness measurements, if empty string it will look for
                        rasterized thk in the input data file
  --mode_opti MODE_OPTI
  --opti_thr_strflowctrl OPTI_THR_STRFLOWCTRL
                        threshold value for strflowctrl
  --opti_init_zero_thk OPTI_INIT_ZERO_THK
                        Initialize the optimization with zero ice thickness
  --opti_regu_param_thk OPTI_REGU_PARAM_THK
                        Regularization weight for the ice thickness in the optimization
  --opti_regu_param_strflowctrl OPTI_REGU_PARAM_STRFLOWCTRL
                        Regularization weight for the strflowctrl field in the optimization
  --opti_smooth_anisotropy_factor OPTI_SMOOTH_ANISOTROPY_FACTOR
                        Smooth anisotropy factor for the ice thickness regularization in the
                        optimization
  --opti_convexity_weight OPTI_CONVEXITY_WEIGHT
                        Convexity weight for the ice thickness regularization in the optimization
  --opti_usurfobs_std OPTI_USURFOBS_STD
                        Confidence/STD of the top ice surface as input data for the optimization
  --opti_strflowctrl_std OPTI_STRFLOWCTRL_STD
                        Confidence/STD of strflowctrl
  --opti_velsurfobs_std OPTI_VELSURFOBS_STD
                        Confidence/STD of the surface ice velocities as input data for the
                        optimization (if 0, velsurfobs_std field must be given)
  --opti_thkobs_std OPTI_THKOBS_STD
                        Confidence/STD of the ice thickness profiles (unless given)
  --opti_divfluxobs_std OPTI_DIVFLUXOBS_STD
                        Confidence/STD of the flux divergence as input data for the optimization
                        (if 0, divfluxobs_std field must be given)
  --opti_control OPTI_CONTROL
                        List of optimized variables for the optimization
  --opti_cost OPTI_COST
                        List of cost components for the optimization
  --opti_nbitmin OPTI_NBITMIN
                        Min iterations for the optimization
  --opti_nbitmax OPTI_NBITMAX
                        Max iterations for the optimization
  --opti_step_size OPTI_STEP_SIZE
                        Step size for the optimization
  --opti_make_holes_in_data OPTI_MAKE_HOLES_IN_DATA
                        This produces artifical holes in data, serve to test the robustness of the
                        method to missing data
  --opti_output_freq OPTI_OUTPUT_FREQ
                        Frequency of the output for the optimization
  --geology_optimized_file GEOLOGY_OPTIMIZED_FILE
                        Geology input file
``` 



### <h1 align="center" id="title"> Code of optimize </h1>


```python 

    def optimize(self):
        """
        This is the optimization routine to invert thk, strflowctrl ans usurf from data
        DEFAULT PARAMETERS ARE
        # nbitmin=50, nbitmax=1000, opti_step_size=0.001,
        # init_zero_thk=True,
        # regu_param_thk=1.0, regu_param_strflowctrl=1.0,
        # smooth_anisotropy_factor=0.2, convexity_weight = 0.002,
        # opti_control=['thk','strflowctrl'], opti_cost=['velsurf','thk'],
        """

        self.initialize_iceflow()

        ###### PERFORM CHECKS PRIOR OPTIMIZATIONS

        # make sure this condition is satisfied
        assert ("usurf" in self.config.opti_cost) == (
            "usurf" in self.config.opti_control
        )

        # make sure the loaded ice flow emulator has these inputs
        assert (
            self.iceflow_mapping["fieldin"]
            == ["thk", "slopsurfx", "slopsurfy", "arrhenius", "slidingco"]
        ) | (
            self.iceflow_mapping["fieldin"]
            == ["thk", "slopsurfx", "slopsurfy", "strflowctrl"]
        )

        # make sure the loaded ice flow emulator has at least these outputs
        assert all(
            [
                (f in self.iceflow_mapping["fieldout"])
                for f in ["ubar", "vbar", "uvelsurf", "vvelsurf"]
            ]
        )

        # make sure that there are lease some profiles in thkobs
        if "thk" in self.config.opti_cost:
            if self.config.thk_profiles_file == "":
                assert not tf.reduce_all(tf.math.is_nan(self.thkobs))

        ###### PREPARE DATA PRIOR OPTIMIZATIONS

        if "thk" in self.config.opti_cost:
            if not self.config.thk_profiles_file == "":
                self.load_thk_profiles()

        if hasattr(self, "uvelsurfobs") & hasattr(self, "vvelsurfobs"):
            self.velsurfobs = tf.stack([self.uvelsurfobs, self.vvelsurfobs], axis=-1)

        if "divfluxobs" in self.config.opti_cost:
            self.divfluxobs = self.smb - self.dhdt

        if not self.config.opti_smooth_anisotropy_factor == 1:
            self.compute_flow_direction_for_anisotropic_smoothing()

        if self.config.opti_make_holes_in_data > 0:
            self.make_data_holes()

        if hasattr(self, "thkinit"):
            self.thk.assign(self.thkinit)
        else:
            self.thk.assign(tf.zeros_like(self.thk))

        if self.config.opti_init_zero_thk:
            self.thk.assign(tf.zeros_like(self.thk))

        ###### PREPARE OPIMIZER

        optimizer = tf.keras.optimizers.Adam(lr=self.config.opti_step_size)

        # initial_learning_rate * decay_rate ^ (step / decay_steps)
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( initial_learning_rate=opti_step_size, decay_steps=100, decay_rate=0.9)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # add scalng for usurf
        self.iceflow_fieldbounds["usurf"] = (
            self.iceflow_fieldbounds["slopsurfx"] * self.dx
        )

        ###### PREPARE VARIABLES TO OPTIMIZE

        if self.iceflow_mapping["fieldin"] == [
            "thk",
            "slopsurfx",
            "slopsurfy",
            "arrhenius",
            "slidingco",
        ]:
            self.iceflow_fieldbounds["strflowctrl"] = (
                self.iceflow_fieldbounds["arrhenius"]
                + self.iceflow_fieldbounds["slidingco"]
            )

        thk = tf.Variable(self.thk / self.iceflow_fieldbounds["thk"])  # normalized vars
        strflowctrl = tf.Variable(
            self.strflowctrl / self.iceflow_fieldbounds["strflowctrl"]
        )  # normalized vars
        usurf = tf.Variable(
            self.usurf / self.iceflow_fieldbounds["usurf"]
        )  # normalized vars

        self.costs = []

        self.tcomp["Optimize"] = []

        # main loop
        for i in range(self.config.opti_nbitmax):

            with tf.GradientTape() as t:

                self.tcomp["Optimize"].append(time.time())

                # is necessary to remember all operation to derive the gradients w.r.t. control variables
                if "thk" in self.config.opti_control:
                    t.watch(thk)
                if "usurf" in self.config.opti_control:
                    t.watch(usurf)
                if "strflowctrl" in self.config.opti_control:
                    t.watch(strflowctrl)

                # update surface gradient
                if (i == 0) | ("usurf" in self.config.opti_control):
                    slopsurfx, slopsurfy = self.compute_gradient_tf(
                        usurf * self.iceflow_fieldbounds["usurf"], self.dx, self.dx
                    )
                    slopsurfx = slopsurfx / self.iceflow_fieldbounds["slopsurfx"]
                    slopsurfy = slopsurfy / self.iceflow_fieldbounds["slopsurfy"]

                if self.iceflow_mapping["fieldin"] == [
                    "thk",
                    "slopsurfx",
                    "slopsurfy",
                    "arrhenius",
                    "slidingco",
                ]:

                    thrv = (
                        self.config.opti_thr_strflowctrl
                        / self.iceflow_fieldbounds["strflowctrl"]
                    )
                    arrhenius = tf.where(strflowctrl <= thrv, strflowctrl, thrv)
                    slidingco = tf.where(strflowctrl <= thrv, 0, strflowctrl - thrv)

                    # build input of the emulator
                    X = tf.concat(
                        [
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.pad(thk, self.PAD, "CONSTANT"), axis=0
                                ),
                                axis=-1,
                            ),
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.pad(slopsurfx, self.PAD, "CONSTANT"), axis=0
                                ),
                                axis=-1,
                            ),
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.pad(slopsurfy, self.PAD, "CONSTANT"), axis=0
                                ),
                                axis=-1,
                            ),
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.pad(arrhenius, self.PAD, "CONSTANT"), axis=0
                                ),
                                axis=-1,
                            ),
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.pad(slidingco, self.PAD, "CONSTANT"), axis=0
                                ),
                                axis=-1,
                            ),
                        ],
                        axis=-1,
                    )

                elif self.iceflow_mapping["fieldin"] == [
                    "thk",
                    "slopsurfx",
                    "slopsurfy",
                    "strflowctrl",
                ]:

                    # build input of the emulator
                    X = tf.concat(
                        [
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.pad(thk, self.PAD, "CONSTANT"), axis=0
                                ),
                                axis=-1,
                            ),
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.pad(slopsurfx, self.PAD, "CONSTANT"), axis=0
                                ),
                                axis=-1,
                            ),
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.pad(slopsurfy, self.PAD, "CONSTANT"), axis=0
                                ),
                                axis=-1,
                            ),
                            tf.expand_dims(
                                tf.expand_dims(
                                    tf.pad(strflowctrl, self.PAD, "CONSTANT"), axis=0
                                ),
                                axis=-1,
                            ),
                        ],
                        axis=-1,
                    )
                else:
                    # ONLY these 2 above cases were implemented !!!
                    sys.exit(
                        "CHANGE THE ICE FLOW EMULATOR -- IMCOMPATIBLE FOR INVERSION "
                    )

                # evalutae th ice flow emulator
                Y = self.iceflow_model(X)

                # get the dimensions of the working array
                Ny, Nx = self.thk.shape

                # save output variables into self.variables for outputs
                for kk, f in enumerate(self.iceflow_mapping["fieldout"]):
                    vars(self)[f].assign(
                        Y[0, :Ny, :Nx, kk] * self.iceflow_fieldbounds[f]
                    )

                # find index of variables in output
                iubar = self.iceflow_mapping["fieldout"].index("ubar")
                ivbar = self.iceflow_mapping["fieldout"].index("vbar")
                iuvsu = self.iceflow_mapping["fieldout"].index("uvelsurf")
                ivvsu = self.iceflow_mapping["fieldout"].index("vvelsurf")

                # save output of the emaultor to compute the costs function
                ubar = (
                    Y[0, :Ny, :Nx, iubar] * self.iceflow_fieldbounds["ubar"]
                )  # NOT normalized vars
                vbar = (
                    Y[0, :Ny, :Nx, ivbar] * self.iceflow_fieldbounds["vbar"]
                )  # NOT normalized vars
                uvelsurf = (
                    Y[0, :Ny, :Nx, iuvsu] * self.iceflow_fieldbounds["uvelsurf"]
                )  # NOT normalized vars
                vvelsurf = (
                    Y[0, :Ny, :Nx, ivvsu] * self.iceflow_fieldbounds["vvelsurf"]
                )  # NOT normalized vars
                velsurf = tf.stack([uvelsurf, vvelsurf], axis=-1)  # NOT normalized vars

                # misfit between surface velocity
                if "velsurf" in self.config.opti_cost:
                    ACT = ~tf.math.is_nan(self.velsurfobs)
                    COST_U = 0.5 * tf.reduce_mean(
                        (
                            (self.velsurfobs[ACT] - velsurf[ACT])
                            / self.config.opti_velsurfobs_std
                        )
                        ** 2
                    )
                else:
                    COST_U = tf.Variable(0.0)

                # misfit between ice thickness profiles
                if "thk" in self.config.opti_cost:
                    if self.config.thk_profiles_file == "":
                        ACT = ~tf.math.is_nan(self.thkobs)
                        COST_H = 0.5 * tf.reduce_mean(
                            (
                                (
                                    self.thkobs[ACT]
                                    - thk[ACT] * self.iceflow_fieldbounds["thk"]
                                )
                                / self.config.opti_thkobs_std
                            )
                            ** 2
                        )
                    else:
                        import tensorflow_addons as tfa

                        COST_H = 0.5 * tf.reduce_mean(
                            (
                                (
                                    self.thktargetprof
                                    - tfa.image.interpolate_bilinear(
                                        tf.expand_dims(
                                            tf.expand_dims(
                                                thk * self.iceflow_fieldbounds["thk"],
                                                axis=0,
                                            ),
                                            axis=-1,
                                        ),
                                        self.jiptprof,
                                        indexing="ij",
                                    )[0, :, 0]
                                )
                                / self.thktargetprof_std
                            )
                            ** 2
                        )
                else:
                    COST_H = tf.Variable(0.0)

                # misfit divergence of the flux
                if ("divfluxobs" in self.config.opti_cost) | (
                    "divfluxfcz" in self.config.opti_cost
                ):

                    divflux = self.compute_divflux(
                        ubar,
                        vbar,
                        thk * self.iceflow_fieldbounds["thk"],
                        self.dx,
                        self.dx,
                    )

                    if "divfluxfcz" in self.config.opti_cost:
                        ACT = self.icemaskobs > 0.5
                        if i % 10 == 0:
                            # his does not need to be comptued any iteration as this is expensive
                            res = stats.linregress(
                                self.usurf[ACT], divflux[ACT]
                            )  # this is a linear regression (usually that's enough)
                        # or you may go for polynomial fit (more gl, but may leads to errors)
                        #  weights = np.polyfit(self.usurf[ACT],divflux[ACT], 2)
                        divfluxtar = tf.where(
                            ACT, res.intercept + res.slope * self.usurf, 0.0
                        )
                    #                        divfluxtar = tf.where(ACT, np.poly1d(weights)(self.usurf) , 0.0 )

                    else:
                        divfluxtar = self.divfluxobs

                    ACT = self.icemaskobs > 0.5
                    COST_D = 0.5 * tf.reduce_mean(
                        (
                            (divfluxtar[ACT] - divflux[ACT])
                            / self.config.opti_divfluxobs_std
                        )
                        ** 2
                    )

                else:
                    COST_D = tf.Variable(0.0)

                # misfit between top ice surfaces
                if "usurf" in self.config.opti_cost:
                    ACT = self.icemaskobs > 0.5
                    COST_S = 0.5 * tf.reduce_mean(
                        (
                            (
                                usurf[ACT] * self.iceflow_fieldbounds["usurf"]
                                - self.usurfobs[ACT]
                            )
                            / self.config.opti_usurfobs_std
                        )
                        ** 2
                    )
                else:
                    COST_S = tf.Variable(0.0)

                # force usurf = usurf - topg
                if "topg" in self.config.opti_cost:
                    ACT = self.icemaskobs == 1
                    COST_T = 10 ** 10 * tf.reduce_mean(
                        (
                            usurf[ACT] * self.iceflow_fieldbounds["usurf"]
                            - thk[ACT] * self.iceflow_fieldbounds["thk"]
                            - self.topg[ACT]
                        )
                        ** 2
                    )
                else:
                    COST_T = tf.Variable(0.0)

                # force zero thikness outisde the mask
                if "icemask" in self.config.opti_cost:
                    COST_O = 10 ** 10 * tf.math.reduce_mean(
                        tf.where(self.icemaskobs > 0.5, 0.0, thk ** 2)
                    )
                else:
                    COST_O = tf.Variable(0.0)

                # Here one enforces non-negative ice thickness, and possibly zero-thickness in user-defined ice-free areas.
                if "thk" in self.config.opti_control:
                    COST_HPO = 10 ** 10 * tf.math.reduce_mean(
                        tf.where(thk >= 0, 0.0, thk ** 2)
                    )
                else:
                    COST_HPO = tf.Variable(0.0)

                # # Make sur to keep reasonable values for strflowctrl
                if "strflowctrl" in self.config.opti_control:
                    COST_STR = 0.5 * tf.reduce_mean(
                        (
                            (
                                strflowctrl * self.iceflow_fieldbounds["strflowctrl"]
                                - self.config.opti_thr_strflowctrl
                            )
                            / self.config.opti_strflowctrl_std
                        )
                        ** 2
                    )
                else:
                    COST_STR = tf.Variable(0.0)

                # Here one adds a regularization terms for the ice thickness to the cost function
                if "thk" in self.config.opti_control:
                    if self.config.opti_smooth_anisotropy_factor == 1:
                        dbdx = thk[:, 1:] - thk[:, :-1]
                        dbdy = thk[1:, :] - thk[:-1, :]
                        REGU_H = self.config.opti_regu_param_thk * (
                            tf.nn.l2_loss(dbdx) + tf.nn.l2_loss(dbdy)
                        )
                    else:
                        dbdx = thk[:, 1:] - thk[:, :-1]
                        dbdx = (dbdx[1:, :] + dbdx[:-1, :]) / 2.0
                        dbdy = thk[1:, :] - thk[:-1, :]
                        dbdy = (dbdy[:, 1:] + dbdy[:, :-1]) / 2.0
                        REGU_H = self.config.opti_regu_param_thk * (
                            tf.nn.l2_loss((dbdx * self.flowdirx + dbdy * self.flowdiry))
                            + self.config.opti_smooth_anisotropy_factor
                            * tf.nn.l2_loss(
                                (dbdx * self.flowdiry - dbdy * self.flowdirx)
                            )
                            - self.config.opti_convexity_weight
                            * tf.math.reduce_sum(thk)
                        )
                else:
                    REGU_H = tf.Variable(0.0)

                # Here one adds a regularization terms for strflowctrl to the cost function
                if "strflowctrl" in self.config.opti_control:
                    dadx = tf.math.abs(strflowctrl[:, 1:] - strflowctrl[:, :-1])
                    dady = tf.math.abs(strflowctrl[1:, :] - strflowctrl[:-1, :])
                    dadx = tf.where(
                        (self.icemaskobs[:, 1:] > 0.5)
                        & (self.icemaskobs[:, :-1] > 0.5),
                        dadx,
                        0.0,
                    )
                    dady = tf.where(
                        (self.icemaskobs[1:, :] > 0.5)
                        & (self.icemaskobs[:-1, :] > 0.5),
                        dady,
                        0.0,
                    )
                    REGU_A = self.config.opti_regu_param_strflowctrl * (
                        tf.nn.l2_loss(dadx) + tf.nn.l2_loss(dady)
                    )
                else:
                    REGU_A = tf.Variable(0.0)

                # sum all component into the main cost function
                COST = (
                    COST_U
                    + COST_H
                    + COST_D
                    + COST_S
                    + COST_T
                    + COST_O
                    + COST_HPO
                    + COST_STR
                    + REGU_H
                    + REGU_A
                )

                vol = (
                    np.sum(thk * self.iceflow_fieldbounds["thk"])
                    * (self.dx ** 2)
                    / 10 ** 9
                )

                if i % self.config.opti_output_freq == 0:
                    print(
                        " OPTI, step %5.0f , ICE_VOL: %7.2f , COST_U: %7.2f , COST_H: %7.2f , COST_D : %7.2f , COST_S : %7.2f , REGU_H : %7.2f , REGU_A : %7.2f "
                        % (
                            i,
                            vol,
                            COST_U.numpy(),
                            COST_H.numpy(),
                            COST_D.numpy(),
                            COST_S.numpy(),
                            REGU_H.numpy(),
                            REGU_A.numpy(),
                        )
                    )

                self.costs.append(
                    [
                        COST_U.numpy(),
                        COST_H.numpy(),
                        COST_D.numpy(),
                        COST_S.numpy(),
                        REGU_H.numpy(),
                        REGU_A.numpy(),
                    ]
                )

                var_to_opti = []
                if "thk" in self.config.opti_control:
                    var_to_opti.append(thk)
                if "usurf" in self.config.opti_control:
                    var_to_opti.append(usurf)
                if "strflowctrl" in self.config.opti_control:
                    var_to_opti.append(strflowctrl)

                # Compute gradient of COST w.r.t. X
                grads = tf.Variable(t.gradient(COST, var_to_opti))

                # this serve to restict the optimization of controls to the mask
                for ii in range(grads.shape[0]):
                    grads[ii].assign(tf.where((self.icemaskobs > 0.5), grads[ii], 0))

                # One step of descent -> this will update input variable X
                optimizer.apply_gradients(
                    zip([grads[i] for i in range(grads.shape[0])], var_to_opti)
                )

                # get back optimized variables in the pool of self.variables
                if "thk" in self.config.opti_control:
                    self.thk.assign(thk * self.iceflow_fieldbounds["thk"])
                    self.thk.assign(tf.where(self.thk < 0.01, 0, self.thk))
                if "strflowctrl" in self.config.opti_control:
                    self.strflowctrl.assign(
                        strflowctrl * self.iceflow_fieldbounds["strflowctrl"]
                    )
                if "usurf" in self.config.opti_control:
                    self.usurf.assign(usurf * self.iceflow_fieldbounds["usurf"])

                self.divflux = self.compute_divflux(
                    self.ubar, self.vbar, self.thk, self.dx, self.dx
                )

                self.compute_rms_std_optimization(i)

                self.tcomp["Optimize"][-1] -= time.time()
                self.tcomp["Optimize"][-1] *= -1

                if i % self.config.opti_output_freq == 0:
                    self.update_plot_inversion(i, self.config.plot_live)
                    self.update_ncdf_optimize(i)
                # self.update_plot_profiles(i,plot_live)

                # stopping criterion: stop if the cost no longer decrease
                # if i>self.config.opti_nbitmin:
                #     cost = [c[0] for c in costs]
                #     if np.mean(cost[-10:])>np.mean(cost[-20:-10]):
                #         break;

        # now that the ice thickness is optimized, we can fix the bed once for all!
        self.topg.assign(self.usurf - self.thk)

        self.output_ncdf_optimize_final()

        self.plot_cost_functions(self.costs, self.config.plot_live)

        np.savetxt(
            os.path.join(self.config.working_dir, "costs.dat"),
            np.stack(self.costs),
            fmt="%.10f",
            header="        COST_U        COST_H      COST_D       COST_S       REGU_H       REGU_A          HPO ",
        )

        np.savetxt(
            os.path.join(self.config.working_dir, "rms_std.dat"),
            np.stack(
                [
                    self.rmsthk,
                    self.stdthk,
                    self.rmsvel,
                    self.stdvel,
                    self.rmsdiv,
                    self.stddiv,
                    self.rmsusurf,
                    self.stdusurf,
                ],
                axis=-1,
            ),
            fmt="%.10f",
            header="        rmsthk      stdthk       rmsvel       stdvel       rmsdiv       stddiv       rmsusurf       stdusurf",
        )

        np.savetxt(
            os.path.join(self.config.working_dir, "strflowctrl.dat"),
            np.array(
                [
                    np.mean(self.strflowctrl[self.icemaskobs > 0.5]),
                    np.std(self.strflowctrl[self.icemaskobs > 0.5]),
                ]
            ),
            fmt="%.3f",
        )

        np.savetxt(
            os.path.join(self.config.working_dir, "volume.dat"),
            np.array([np.sum(self.thk) * self.dx * self.dx / (10 ** 9)]),
            fmt="%.3f",
        )

        np.savetxt(
            os.path.join(self.config.working_dir, "tcompoptimize.dat"),
            np.array([np.sum([f for f in self.tcomp["Optimize"]])]),
            fmt="%.3f",
        )

``` 

