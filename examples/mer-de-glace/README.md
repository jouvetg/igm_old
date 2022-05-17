
# Overview

This set-up permits to model the mer de glace glacier (France) with i) one step of data assimilation (inverse modelling) and ii) one step of forward modelling based on a modification of the inferred ELA. 

# Installing Python Packages

Follow the section "Installing Python Packages" on https://github.com/jouvetg/igm 

# Data

- 'observation-RGI-3642.nc' contains the raster data including ice surface veolocities (Millan and al. 2022), initial ice thickness (Millan and al. 2022), ice mask (Linsbauer and al. 2021), ice surface elevations (swissALTI3D).

# Code

- 'igm-run-inverse.py' for the inverse modelling / diagonostic modelling, it will create 'geology.nc' needed for the following
- 'igm-run-forward.py' for the forward modelling / prognostic modelling

# Ice flow emulator

... is defined in the folder f12_cfsflow or f17_pismbp_GJ_22_a

# Usage 
 
		
Make sure the IGM's dependent libraries are installed, or activate your igm environment with conda

		conda activate igm
	 
Then run

		python igm-run-inverse.py
		python igm-run-forward.py
		
You may clean behind you after experimenting:

		sh clean.sh

# References (method)

	@article{IGM-inv,
	  author       = "G. Jouvet",  
	  title        = "Inversion of a Stokes ice flow model emulated by deep learning",
	  journal      = "Journal of Glaciology",
	  year         = 2022,
	}

	@article{IGM,
	  author       = "G. Jouvet, G. Cordonnier, B. Kim, M. Luethi, A. Vieli, A. Aschwanden",  
	  title        = "Deep learning speeds up ice flow modelling by several orders of magnitude",
	  journal      = "Journal of Glaciology",
	  year         = 2021,
	}
	
# References (data)

swissALTI3D - SwissTopo - admin.ch

	@article{millan2019mapping,
	  title={Mapping surface flow velocity of glaciers at regional scale using a multiple sensors approach},
	  author={Millan, Romain and Mouginot, J{\'e}r{\'e}mie and Rabatel, Antoine and Jeong, Seongsu and Cusicanqui, Diego and Derkacheva, Anna and Chekki, Mondher},
	  journal={Remote Sensing},
	  volume={11},
	  number={21},
	  pages={2498},
	  year={2019},
	  publisher={Multidisciplinary Digital Publishing Institute}
	}

	@article{linsbauer2021new,
	  title={The new Swiss Glacier Inventory SGI2016: From a topographical to a glaciological dataset},
	  author={Linsbauer, Andreas and Huss, Matthias and Hodel, Elias and Bauder, Andreas and Fischer, Mauro and Weidmann, Yvo and B{\"a}rtschi, Hans and Schmassmann, Emanuel},
	  journal={Frontiers in Earth Science},
	  pages={774},
	  year={2021},
	  publisher={Frontiers}
	}


