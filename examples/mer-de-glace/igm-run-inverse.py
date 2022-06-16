# import basic libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys 

# import igm libary
from igm import *

# create an object igm, which contains variable and functions
igm=igm()

# define working directory and input files
igm.config.working_dir            = ''                     
igm.config.iceflow_model_lib_path = '../../model-lib/f17_pismbp_GJ_22_a/' 
igm.config.observation_file       = 'observation-RGI-3642.nc' # this is the main input file
igm.config.plot_result           = False
igm.config.plot_live             = False
 
igm.config.opti_output_freq       =  50     # Frequency for output
igm.config.opti_nbitmax           = 500     # Number of iterations for the optimization
igm.config.thk_profiles_file      = ''     

igm.config.opti_usurfobs_std             = 2.0   # Tol to fit top ice surface
igm.config.opti_velsurfobs_std           = 3.0   # Tol to fit surface speeds
igm.config.opti_thkobs_std               = 5.0   # Tol to fit ice thk profiles
igm.config.opti_strflowctrl_std          = 5.0   # Tol to fit strflowctr
igm.config.opti_divfluxobs_std           = 0.5   # Tol to fit the flux divergence (NON DEFAULT)

igm.config.opti_regu_param_thk           = 10.0  # weight for the regul. of thk
igm.config.opti_smooth_anisotropy_factor = 0.2   # Smooth anisotropy factor
igm.config.opti_regu_param_strflowctrl   = 5.0   # weight for the regul. of strflowctrl (NON DEFAULT)

igm.config.opti_step_size                = 0.005 # this is the step size while optimiting

igm.config.opti_init_zero_thk    = False 
igm.config.opti_convexity_weight = 0.0   # this can be set to zero when thk not initalized to zero

# This combination works when there exist observed ice thickness data
# As we do not use any ice thk profiles, we assume A=78 
igm.config.opti_control = ['thk','usurf'] # ,'strflowctrl'
igm.config.opti_cost    = ['velsurf','icemask','usurf','divfluxfcz'] # 'thk' ,'thk'

igm.initialize()
with tf.device(igm.device_name):
    igm.load_ncdf_data(igm.config.observation_file)
    igm.initialize_fields()

    # smooth the mask fo technical reasons
    from scipy.ndimage import gaussian_filter
    igm.icemaskobs.assign( gaussian_filter(igm.icemaskobs, 2, mode="constant")>0.4 )

    igm.optimize()
    
igm.plot_opti_diff('optimize.nc','thk', plot_live=False)
igm.plot_opti_diff('optimize.nc','velsurf_mag', plot_live=False)
 
################################################################
################################################################
################################################################

#plt.imshow(igm.divflux,origin='lower',vmin=-10,vmax=10)
ACT = (igm.thk>1.0)&(np.abs(igm.divflux)>0.1)                        
res = stats.linregress(igm.usurf[ACT],igm.divflux[ACT])  
ELA = - res.intercept/res.slope
gradmb = res.slope

fig, ax = plt.subplots(1,1,figsize=(6,6),dpi=200) 
plt.imshow(igm.divflux,origin='lower',vmin=-10,vmax=10) ; plt.colorbar()
ax.axis('off')
plt.savefig( 'divflux.png', pad_inches=0 )
print('ELA :',ELA,' gradmb :',gradmb)
