
### <h1 align="center" id="title">Inverse modelling (data assimilation) with IGM </h1>

A data assimilation module of IGM permits to seek optimal ice thickness, top ice surface, and ice flow parametrization (red variables in the following figure), that best explains observational data such as surface ice speeds, ice thickness local profiles, top ice surface (blue variables in the following figure) while being consistent with the ice flow emulator. This page explain how to use the data assimilation module as a preliminary step in IGM of a forward/prognostic model run.

![](https://github.com/jouvetg/igm/blob/main/fig/scheme_simple_invert.png)

# Getting the data 

The first thing you need to do is to get as much data as possible, this includes:

* Observed surface ice velocities ${\bf u}^{s,obs}$, e.g. from Millan and al. (2022).
* Surface top elevation $s^{obs}$, e.g. SRTM, ESA GLO-30, ...
* Ice thickness profiles $h_p^{obs}$, e.g. GlaThiDa
* Glacier outlines, and resulting mask, e.g. from the Randolph Glacier Inventory.

Of course, you may not have all these data, which is fine. It is possible to keep working with a reduce amount of data, but we will have make assumptions to reduce the number of variables to optimize (controls) to keep the optimization problem well-posed (with a unique solution).

All the data need to be assemblied in 2D raster grid in an netcdf observation.nc file using convention variable names but ending with 'obs'. E.g. observation.nc contains fields 'usurfobs' (observed top surface elevation), thkobs (observed thickness profiles, use nan or novalue where no data is available), icemaskobs (this mask from RGI outline serve to enforce zero ice thickness outside the mask), uvelsurfobs and vvelsurfobs (x- and y- components of the horizontal surface ice velocity, use nan or novalue where no data is available), thkinit (this is a formerly inferred ice thickness field that may be used to initalize the inverse model, otherwise it would start from thk=0).

# Asumption on the ice flow control

Optimizing for both Arrhenius factor ($A$) and sliding coefficient ($c$) would lead to multiple solutions as several combination of the two may explain the observed ice flow similarly. To deal with this issue, we introduce a single control of the ice flow strenght (named as strflowctrl in IGM) $\tilde{A} = A + \lambda c$, where $A$ is the Arrhenius factor that controls the ice shearing from cold-ice case (low $A$) to temperate ice case ($A=78$ MPa$^{-3}$ a$^{-1}$), $c$ is a sliding coefficient that controls the strength of basal motion from no sliding ($c=0$) to high sliding (high $c$) and $\lambda=1$ km$^{-1}$ is a given parameter. 

![](https://github.com/jouvetg/igm/blob/main/fig/strflowctrl.png)

# Set-up the inverse model (cost function to minimize)

The optimization problem consists of finding spatially varying fields ($h$, $\tilde{A}$, $s$) that minimize the cost function
$$ \mathcal{J}(h,\tilde{A},s) = \mathcal{C}^u + \mathcal{C}^h + \mathcal{C}^s + \mathcal{C}^{d} + \mathcal{R}^h +  \mathcal{R}^{\tilde{A}}, $$

where $\mathcal{C}^u$ is the misfit between modeled and observed surface ice velocities ($\mathcal{F}$ is the output of the ice flow emulator/neural network):
$$ \mathcal{C}^u = \int_{\Omega} \frac{1}{2 \sigma_u^2} \left| {\bf u}^{s,obs} - \mathcal{F}( h, \frac{\partial s}{\partial x}, \frac{\partial s}{\partial y}, \tilde{A})  \right|^2,  $$

where $\mathcal{C}^h$ is the misfit between modeled and observed ice thickness profiles:
$$ \mathcal{C}^h = \sum_{p=1,...,P} \sum_{i=1,...,M_p} \frac{1}{2 \sigma_h^2}  | h_p^{obs}  (x^p_i, y^p_i) - h (x^p_i, y^p_i) |^2, $$

where $\mathcal{C}^s$ is the misfit between the modeled and observed top ice surface:
$$ \mathcal{C}^s = \int_{\Omega} \frac{1}{2 \sigma_s^2}  \left| s - s^{obs}  \right|^2,$$

where $\mathcal{C}^{d}$ is a misfit term between the flux divergence $\nabla \cdot (h {\bar{\bf u}})$ and its polynomial 
regression $d$ with respect to the ice surface elevation $s(x,y)$ to enforce smoothness with  dependence to $s$:
$$ \mathcal{C}^{d} = \int_{\Omega} \frac{1}{2 \sigma_d^2} \left| \nabla \cdot (h {\bar{\bf u}}) - d  \right|^2, $$

where $\mathcal{R}^h$ is a regularization term to enforce anisotropic smoothness and convexity of $h$:
$$ \mathcal{R}^h = \alpha_h \int_{h>0} \left(  | \nabla h \cdot \tilde{{\bf u}}^{s,obs} |^2 + \beta  | \nabla h \cdot (\tilde{{\bf u}}^{s,obs})^{\perp} |^2   -  \gamma h  \right)  $$

where the last term is a regularization term to enforce smooth $\tilde{A}$:
$$ \mathcal{R}^{\tilde{A}} = \alpha_{\tilde{A}} \int_{\Omega} | \nabla  \tilde{A}  |^2. $$

The above optimization problem is the most general case, however, you may select only some components.
For that, you need to define 

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
