
### <h1 align="center" id="title">Inverse modelling (data assimilation) with IGM </h1>

A data assimilation module of IGM permits to seek optimal ice thickness, top ice surface, and ice flow parametrization (red variables in the following figure), that best explains observational data such as surface ice speeds, ice thickness local profiles, top ice surface (blue variables in the following figure) while being consistent with the ice flow emulator. This page explain how to use the data assimilation module as a preliminary step in IGM of a forward/prognostic model run.

![](https://github.com/jouvetg/igm/blob/main/fig/scheme_simple_invert.png)

# Step 1: Getting the data 

The first thing you need to do is to get as much data as possible, this includes:

* Observed surface ice velocities ${\bf u}^{s,obs}$, e.g. from Millan and al. (2022).
* Surface top elevation $s^{obs}$, e.g. SRTM, ESA GLO-30, ...
* Ice thickness profiles $h_p^{obs}$, e.g. GlaThiDa
* Glacier outlines, and resulting mask, e.g. from the Randolph Glacier Inventory.

Of course, you may not have all these data, which is fine. It is possible to keep working with a reduce amount of data, but we will have make assumptions to reduce the number of variables to optimize (controls) to keep the optimization problem well-posed (with a unique solution).

All the data need to be assemblied in 2D raster grid in an netcdf observation.nc file using convention variable names but ending with 'obs'. E.g. observation.nc contains fields 'usurfobs' (observed top surface elevation), thkobs (observed thickness profiles, use nan or novalue where no data is available), icemaskobs (this mask from RGI outline serve to enforce zero ice thickness outside the mask), uvelsurfobs and vvelsurfobs (x- and y- components of the horizontal surface ice velocity, use nan or novalue where no data is available), thkinit (this is a formerly inferred ice thickness field that may be used to initalize the inverse model, otherwise it would start from thk=0).

# Step 2: Set-up the inverse model (cost function to minimize)

The optimization problem consists of finding spatially varying fields $h$, $\tilde{A}$ and $s$ that minimize the cost function
$$ \mathcal{J}(h,\tilde{A},s) = \mathcal{C}^u + \mathcal{C}^h + \mathcal{C}^s + \mathcal{C}^{d} + \mathcal{R}^h +  \mathcal{R}^{\tilde{A}} $$
where
$$ \mathcal{C}^u = \int_{\Omega} \frac{1}{2 \sigma_u^2} \left| {\bf u}^{s,obs} - \mathcal{F}( h, \frac{\partial s}{\partial x}, \frac{\partial s}{\partial y}, \tilde{A})  \right|^2  $$
is the misfit between modeled and observed surface ice velocities ($\mathcal{F}$ is the output of the ice flow emulator/neural network),
$$ \mathcal{C}^h = \sum_{p=1,...,P} \sum_{i=1,...,M_p} \frac{1}{2 \sigma_h^2}  | h_p^{obs}  (x^p_i, y^p_i) - h (x^p_i, y^p_i) |^2 $$
is the misfit between modeled and observed ice thickness profiles,
$$ \mathcal{C}^s = \int_{\Omega} \frac{1}{2 \sigma_s^2}  \left| s - s^{obs}  \right|^2 $$
is the misfit between the modeled and observed top ice surface,
$$ \mathcal{C}^{d} = \int_{\Omega} \frac{1}{2 \sigma_d^2} \left| \nabla \cdot (h {\bar{\bf u}}) - d^{poly}  \right|^2, $$
is a misfit term between the flux divergence $\nabla \cdot (h {\bar{\bf u}})$ and its polynomial regression 
$d^{poly}$ with respect to the ice surface elevation $s(x,y)$ to enforce smoothness with linear dependence to $s$, 
$$ \mathcal{R}^h = \alpha_h \int_{h>0} \left(  | \nabla h \cdot \tilde{{\bf u}}^{s,obs} |^2 
+ \beta  | \nabla h \cdot (\tilde{{\bf u}}^{s,obs})^{\perp} |^2   -    \gamma h  \right)  $$
is a regularization term to enforce anisotropic smoothness and convexity of $h$ (see next paragraph),
$$ \mathcal{R}^{\tilde{A}} = \alpha_{\tilde{A}} \int_{\Omega} | \nabla  \tilde{A}  |^2  $$
is a regularization term to enforce smooth $\tilde{A}$.

This sentence uses `$` delimiters to show math inline:  $\sqrt{3x-1}+(1+x)^2$

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
