



### <h1 align="center" id="title"> Parameters of base_parameters </h1>


``` 

usage: make-doc-function-md.py [-h] [--working_dir WORKING_DIR] [--geology_file GEOLOGY_FILE]
                               [--resample RESAMPLE] [--tstart TSTART] [--tend TEND]
                               [--restartingfile RESTARTINGFILE] [--verbosity VERBOSITY]
                               [--tsave TSAVE] [--plot_result PLOT_RESULT] [--plot_live PLOT_LIVE]
                               [--usegpu USEGPU] [--stop STOP]
                               [--init_strflowctrl INIT_STRFLOWCTRL]
                               [--init_slidingco INIT_SLIDINGCO] [--init_arrhenius INIT_ARRHENIUS]
                               [--vel3d_active VEL3D_ACTIVE] [--dz DZ] [--maxthk MAXTHK]
                               [--clim_update_freq CLIM_UPDATE_FREQ] [--type_climate TYPE_CLIMATE]
                               [--erosion_include EROSION_INCLUDE] [--erosion_cst EROSION_CST]
                               [--erosion_exp EROSION_EXP]
                               [--erosion_update_freq EROSION_UPDATE_FREQ]
                               [--uplift_include UPLIFT_INCLUDE] [--uplift_rate UPLIFT_RATE]
                               [--uplift_update_freq UPLIFT_UPDATE_FREQ]
                               [--iceflow_model_lib_path ICEFLOW_MODEL_LIB_PATH]
                               [--multiple_window_size MULTIPLE_WINDOW_SIZE]
                               [--force_max_velbar FORCE_MAX_VELBAR] [--optimize OPTIMIZE]
                               [--opti_vars_to_save OPTI_VARS_TO_SAVE]
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
                               [--vars_to_save VARS_TO_SAVE] [--varplot VARPLOT]
                               [--varplot_max VARPLOT_MAX] [--mb_update_freq MB_UPDATE_FREQ]
                               [--type_mass_balance TYPE_MASS_BALANCE] [--mb_scaling MB_SCALING]
                               [--mb_simple_file MB_SIMPLE_FILE]
                               [--smb_model_lib_path SMB_MODEL_LIB_PATH] [--cfl CFL]
                               [--dtmax DTMAX] [--tracking_particles TRACKING_PARTICLES]
                               [--frequency_seeding FREQUENCY_SEEDING]
                               [--density_seeding DENSITY_SEEDING]

optional arguments:
  -h, --help            show this help message and exit
  --working_dir WORKING_DIR
                        Working directory (default empty string)
  --geology_file GEOLOGY_FILE
                        Geology input file (default: geology.nc)
  --resample RESAMPLE   Resample the data of ncdf data file to a coarser resolution (default: 1)
  --tstart TSTART       Start modelling time (default 0)
  --tend TEND           End modelling time (default: 100)
  --restartingfile RESTARTINGFILE
                        Provide restarting file if no empty string (default: empty string)
  --verbosity VERBOSITY
                        Verbosity level of IGM (default: 0 = no verbosity)
  --tsave TSAVE         Save result each X years (default: 10)
  --plot_result PLOT_RESULT
                        Save plot results in png (default: False)
  --plot_live PLOT_LIVE
                        Display plots live the results during computation (Default: False)
  --usegpu USEGPU       Use the GPU instead of CPU (default: True)
  --stop STOP           experimental, just to get fair comp time, to be removed ....
  --init_strflowctrl INIT_STRFLOWCTRL
                        Initial strflowctrl (default 78)
  --init_slidingco INIT_SLIDINGCO
                        Initial sliding coeeficient slidingco (default: 0)
  --init_arrhenius INIT_ARRHENIUS
                        Initial arrhenius factor arrhenuis (default: 78)
  --vel3d_active VEL3D_ACTIVE
                        Is the computational of the 3D vel active?
  --dz DZ               Vertical discretization constant spacing
  --maxthk MAXTHK       Vertical maximum thickness
  --clim_update_freq CLIM_UPDATE_FREQ
                        Update the climate each X years (default: 1)
  --type_climate TYPE_CLIMATE
                        This keywork serves to identify & call the climate forcing. If an empty
                        string, this function is not called (Default: )
  --erosion_include EROSION_INCLUDE
                        Include a model for bedrock erosion (Default: False)
  --erosion_cst EROSION_CST
                        Erosion multiplicative factor, here taken from Herman, F. et al. Erosion by
                        an Alpine glacier. Science 350, 193–195 (2015)
  --erosion_exp EROSION_EXP
                        Erosion exponent factor, here taken from Herman, F. et al. Erosion by an
                        Alpine glacier. Science 350, 193–195 (2015)
  --erosion_update_freq EROSION_UPDATE_FREQ
                        Update the erosion only each X years (Default: 100)
  --uplift_include UPLIFT_INCLUDE
                        Include a model with constant bedrock uplift
  --uplift_rate UPLIFT_RATE
                        Uplift rate in m/y (default 2 mm/y)
  --uplift_update_freq UPLIFT_UPDATE_FREQ
                        Update the uplift only each X years (Default: 100 years)
  --iceflow_model_lib_path ICEFLOW_MODEL_LIB_PATH
                        Directory path of the deep-learning ice flow model
  --multiple_window_size MULTIPLE_WINDOW_SIZE
                        In case the ANN is a U-net, it must force window size to be multiple of
                        e.g. 8 (default: 0)
  --force_max_velbar FORCE_MAX_VELBAR
                        This permits to artificially upper-bound velocities, active if > 0
                        (default: 0)
  --optimize OPTIMIZE   Do the data ssimilation (inverse modelling) prior forward modelling
                        (Default: False)
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
  --vars_to_save VARS_TO_SAVE
                        List of variables to be recorded in the ncdf file
  --varplot VARPLOT     variable to plot
  --varplot_max VARPLOT_MAX
                        maximum value of the varplot variable used to adjust the scaling of the
                        colorbar
  --mb_update_freq MB_UPDATE_FREQ
                        Update the mass balance each X years (1)
  --type_mass_balance TYPE_MASS_BALANCE
                        This keywork permits to identify the type of mass balance model, can be:
                        zero, simple, nn or given (Default: simple)
  --mb_scaling MB_SCALING
                        The paramter permit to make a simple mass balance scaling
  --mb_simple_file MB_SIMPLE_FILE
                        Name of the imput file for the simple mass balance model
  --smb_model_lib_path SMB_MODEL_LIB_PATH
                        Model directory in case the smb model in use is 'nn', i.e. neural network
  --cfl CFL             CFL number for the stability of the mass conservation scheme, it must be
                        below 1 (Default: 0.3)
  --dtmax DTMAX         Maximum time step allowed, used only with slow ice (default: 10.0)
  --tracking_particles TRACKING_PARTICLES
                        Is the particle tracking active? (Default: False)
  --frequency_seeding FREQUENCY_SEEDING
                        Frequency of seeding (default: 10)
  --density_seeding DENSITY_SEEDING
                        Density of seeding (default: 0.2)
``` 

