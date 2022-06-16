
### <h1 align="center" id="title">Inverse modelling (data assimilation) with IGM </h1>

A data assimilation module of IGM permits to seek optimal ice thickness, top ice surface, and ice flow parametrization (red variables in the following figure), that best explains observational data such as surface ice speeds, ice thickness profiles, top ice surface (blue variables in the following figure) while being consistent with the ice flow emulator used in forwrd modelling. This page explains how to use the data assimilation module as a preliminary step in IGM of a forward/prognostic model run. For methodological details, please look at the paper reference given below.

![](https://github.com/jouvetg/igm/blob/main/fig/scheme_simple_invert.png)

# Getting the data 

The first thing you need to do is to get as much data as possible, this includes:

* Observed surface ice velocities ${\bf u}^{s,obs}$, e.g. from Millan and al. (2022).
* Surface top elevation $s^{obs}$, e.g. SRTM, ESA GLO-30, ...
* Ice thickness profiles $h_p^{obs}$, e.g. GlaThiDa
* Glacier outlines, and resulting mask, e.g. from the Randolph Glacier Inventory.

Of course, you may not have all these data, which is fine. You work with a reduced amount of data, however, you will have make assumptions to reduce the number of variables to optimize (controls) to keep the optimization problem well-posed (i.e., with a unique solution).

All the data need to be assemblied in 2D raster grid in an netcdf file (called by default observation.nc) using convention variable names but ending with 'obs'. E.g. observation.nc contains fields 'usurfobs' (observed top surface elevation), 'thkobs' (observed thickness profiles, use nan or novalue where no data is available), 'icemaskobs' (this mask from RGI outline serve to enforce zero ice thickness outside the mask), 'uvelsurfobs' and 'vvelsurfobs' (x- and y- components of the horizontal surface ice velocity, use nan or novalue where no data is available), 'thkinit' (this may be a formerly-inferred ice thickness field to initalize the inverse model, otherwise it would start from thk=0).

# Asumption on the ice flow control

Optimizing for both Arrhenius factor (A) and sliding coefficient (c) would lead to multiple solutions as several combinations of the two may explain the observed ice flow similarly. To deal with this issue, we introduce a single control of the ice flow strenght (named as 'strflowctrl' in IGM) $\tilde{A}$ = A + lambda c, where A is the Arrhenius factor that controls the ice shearing from cold-ice case (low A) to temperate ice case (A=78), c is a sliding coefficient that controls the strength of basal motion from no sliding (c=0) to high sliding (high c) and lambda=1 is a given parameter. 

![](https://github.com/jouvetg/igm/blob/main/fig/strflowctrl.png)

# General optimization setting

The optimization problem consists of finding spatially varying fields ($h$, $\tilde{A}$, $s$) that minimize the cost function
$$ \mathcal{J}(h,\tilde{A},s) = \mathcal{C}^u + \mathcal{C}^h + \mathcal{C}^s + \mathcal{C}^{d} + \mathcal{R}^h +  \mathcal{R}^{\tilde{A}} +  \mathcal{P}^h, $$

where $\mathcal{C}^u$ is the misfit between modeled and observed surface ice velocities ($\mathcal{F}$ is the output of the ice flow emulator/neural network):
$$ \mathcal{C}^u = \int_{\Omega} \frac{1}{2 \sigma_u^2} \left| {\bf u}^{s,obs} - \mathcal{F}( h, \frac{\partial s}{\partial x}, \frac{\partial s}{\partial y}, \tilde{A})  \right|^2,  $$

where $\mathcal{C}^h$ is the misfit between modeled and observed ice thickness profiles:
$$ \mathcal{C}^h = \sum_{p=1,...,P} \sum_{i=1,...,M_p} \frac{1}{2 \sigma_h^2}  | h_p^{obs}  (x^p_i, y^p_i) - h (x^p_i, y^p_i) |^2, $$

where $\mathcal{C}^s$ is the misfit between the modeled and observed top ice surface:
$$ \mathcal{C}^s = \int_{\Omega} \frac{1}{2 \sigma_s^2}  \left| s - s^{obs}  \right|^2,$$

where $\mathcal{C}^{d}$ is a misfit term between the flux divergence and its polynomial 
regression $d$ with respect to the ice surface elevation $s(x,y)$ to enforce smoothness with  dependence to $s$:
$$ \mathcal{C}^{d} = \int_{\Omega} \frac{1}{2 \sigma_d^2} \left| \nabla \cdot (h {\bar{\bf u}}) - d  \right|^2, $$

where $\mathcal{R}^h$ is a regularization term to enforce anisotropic smoothness and convexity of $h$:
$$ \mathcal{R}^h = \alpha_h \int_{h>0} \left(  | \nabla h \cdot \tilde{{\bf u}}^{s,obs} |^2 + \beta  | \nabla h \cdot (\tilde{{\bf u}}^{s,obs})^{\perp} |^2   -  \gamma h  \right),  $$

where $\mathcal{R}^{\tilde{A}}$ is a regularization term to enforce smooth A:
$$ \mathcal{R}^{\tilde{A}} = \alpha_{\tilde{A}} \int_{\Omega} | \nabla  \tilde{A}  |^2, $$

where $\mathcal{P}^h$ is a penalty term to enforce nonnegative ice thickness, and zero thickness outside a given mask:
$$ \mathcal{P}^h  = 10^{10} \times \left( \int_{h<0} h^2 + \int_{\mathcal{M}^{\rm ice-free}} h^2 \right).$$

# Define controls and cost components

The above optimization problem is the most general case, however, you may select only some components as follows: 

* the list of control variables you wish to optimize, e.g.
```python
igm.config.opti_control=['thk','strflowctrl','usurf'] # this is the most general case  
igm.config.opti_control=['thk','usurf'] # this will only optimze ice thickness and top surface elevation
igm.config.opti_control=['thk'] # this will only optimze ice thickness 
```
* the list of cost components you wish to minimize, e.g.
```python
igm.config.opti_cost=['velsurf','thk','usurf','divfluxfcz','icemask']  # this is the most general case  
igm.config.opti_cost=['velsurf','icemask']  # In this case, you only fit surface velocity and ice mask.
```
Make sure you have a balance between controls and constraints to ensure the problem to have a unique solution.

# Exploring parameters

There are quite a lot of parameters that may need to be tuned for each applications. 

First, you may change confidence levels $\sigma^u, \sigma^h, \sigma^s, \sigma^d$ to fit surface ice velocity, ice thickness, surface top elevation, or divergence of the flux as follows:

```python
igm.config.opti_velsurfobs_std = 5 # unit m/y
igm.config.opti_thkobs_std     = 5 # unit m
igm.config.opti_usurfobs_std   = 5 # unit m
igm.config.opti_divfluxobs_std = 1 # unit m/y
```

Second, you may change regularization terms such as $\alpha^h, \alpha^A$, which control the weight of regularizations for the ice thickness and strflowctrl, or beta and gamma, which controls the smoothing anisotropy (we force further smoothness along the flow than across flow) and convexity as follows:

```python 
igm.config.opti_regu_param_thk = 10.0            # weight for the regul. of thk
igm.config.opti_regu_param_strflowctrl = 1.0     # weight for the regul. of strflowctrl
igm.config.opti_smooth_anisotropy_factor = 0.2
igm.config.opti_convexity_weight = 0.002
```

Hereabove, increasing $\alpha^h, \alpha^A$ will make h and A smoother. Taking beta=1 occurs to enforce isotropic smoothing, reducing beta will make the smoothing more and more anisotropic to enforce further smoothing along ice flow direction than accross directions (as expected for the topography of a glacier bedrock, which has been submitted to glacier erosion for long times). Lastly, setting parameter gamma to a small value brings a bit of convexity in the system, this may help when initializing the inverse modelled with zero thickness, or to treat rmargin egions with hardly no data available.

Lastly, there are a couple of other parameters we may be interest to change

```python 
igm.config.opti_nbitmax       = 1000   # Number of it. for the optimization
igm.config.opti_step_size     = 0.001  # step size in the optimization iterative algorithm
igm.config.opti_init_zero_thk = True   # Force inializing with zero ice thickness (otherwise take thkinit)
```

# Running the optimization

The optimization scheme is implemented in igm function optimize(), calling it for inverse modelling would look like this:

```python 
import numpy as np
import tensorflow as tf

import igm

igm = igm() 
 
# change parameters
igm.config.iceflow_model_lib_path='../../model-lib/f17_pismbp_GJ_22_a' 
igm.config.opti_control=['thk','strflowctrl','usurf']
igm.config.opti_cost=['velsurf','thk','usurf','divfluxfcz','icemask']   
igm.config.opti_usurfobs_std             = 5.0   # Tol to fit top ice surface 

igm.initialize()

with tf.device(igm.device_name):
    igm.load_ncdf_data(igm.config.observation_file)
    igm.initialize_fields()
    igm.initialize_iceflow()
    igm.optimize()
    
igm.print_all_comp_info()
```

# Reference

	@article{IGM-inv,
	  author       = "Jouvet, G.",
	  title        = "Inversion of a Stokes ice flow model emulated by deep learning",
	  DOI          = "10.1017/jog.2022.41",
	  journal      = "Journal of Glaciology",
	  year         = "2022",
	  pages        = "1--14",
	  publisher    = "Cambridge University Press"
	}
