# import basic libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys 

# import igm libary
from igm import Igm

# create an object igm, which contains variable and functions
glacier=Igm()

# define working directory and input files
glacier.config.working_dir            = ''                     
glacier.config.iceflow_model_lib_path = '../../model-lib/f17_pismbp_GJ_22_a/' 
glacier.config.observation_file       = 'observation-RGI-3642.nc' # this is the main input file
glacier.config.plot_result           = False
glacier.config.plot_live             = False
 
glacier.config.opti_output_freq       =  50     # Frequency for output
glacier.config.opti_nbitmax           = 500     # Number of iterations for the optimization
glacier.config.thk_profiles_file      = ''     

glacier.config.opti_usurfobs_std             = 2.0   # Tol to fit top ice surface
glacier.config.opti_velsurfobs_std           = 3.0   # Tol to fit surface speeds
glacier.config.opti_thkobs_std               = 5.0   # Tol to fit ice thk profiles
glacier.config.opti_strflowctrl_std          = 5.0   # Tol to fit strflowctr
glacier.config.opti_divfluxobs_std           = 0.5   # Tol to fit the flux divergence (NON DEFAULT)

glacier.config.opti_regu_param_thk           = 10.0  # weight for the regul. of thk
glacier.config.opti_smooth_anisotropy_factor = 0.2   # Smooth anisotropy factor
glacier.config.opti_regu_param_strflowctrl   = 5.0   # weight for the regul. of strflowctrl (NON DEFAULT)

glacier.config.opti_step_size                = 0.005 # this is the step size while optimiting

glacier.config.opti_init_zero_thk    = False 
glacier.config.opti_convexity_weight = 0.0   # this can be set to zero when thk not initalized to zero

# This combination works when there exist observed ice thickness data
# As we do not use any ice thk profiles, we assume A=78 
glacier.config.opti_control = ['thk','usurf'] # ,'strflowctrl'
glacier.config.opti_cost    = ['velsurf','icemask','usurf','divfluxfcz'] # 'thk' ,'thk'

glacier.initialize()
with tf.device(glacier.device_name):
    glacier.load_ncdf_data(glacier.config.observation_file)
    glacier.initialize_fields()

    # smooth the mask fo technical reasons
    from scipy.ndimage import gaussian_filter
    glacier.icemaskobs.assign( gaussian_filter(glacier.icemaskobs, 2, mode="constant")>0.4 )

    glacier.optimize()
    
glacier.plot_opti_diff('optimize.nc','thk', plot_live=False)
glacier.plot_opti_diff('optimize.nc','velsurf_mag', plot_live=False)
 
################################################################
################################################################
################################################################

from scipy import stats

#plt.imshow(glacier.divflux,origin='lower',vmin=-10,vmax=10)
ACT = (glacier.thk>1.0)&(np.abs(glacier.divflux)>0.1)                        
res = stats.linregress(glacier.usurf[ACT],glacier.divflux[ACT])  
ELA = - res.intercept/res.slope
gradmb = res.slope

fig, ax = plt.subplots(1,1,figsize=(6,6),dpi=200) 
plt.imshow(glacier.divflux,origin='lower',vmin=-10,vmax=10) ; plt.colorbar()
ax.axis('off')
plt.savefig( 'divflux.png', pad_inches=0 )
print('ELA :',ELA,' gradmb :',gradmb)
