
# Overview

This set-up permits to test 5 different optimization schemes for inverting an emulated ice flow model, i.e. to seek for optimal ice thickness, top ice surface, and ice flow parametrization, that best explain observational data while being consistent with the ice flow emulator. 

# Installing Python Packages

Follow the section "Installing Python Packages" on https://github.com/jouvetg/igm 

# Data

- 'observation.nc' contains the raster data including ice surface veolocities, initial ice thickness, ice thickness profiles, ice mask, ice surface elevations. For the origin of data, please check below or at the metadat of the netcdf file.

# Code

- 'igm-run.py' is the main python script to run for testing

# Ice flow emulator

... is defined in the folder f12_cfsflow or f17_pismbp_GJ_22_a

# Usage 
		
Make sure the IGM's dependent libraries are installed, or activate your igm environment with conda

		conda activate igm
	 
Then run

		python igm-run.py
		
and try different optimization schemes by changing the controls variables and the components of the cost function directly in igm-run.py.
		
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


	@article{grab2021ice,
	  title={Ice thickness distribution of all Swiss glaciers based on extended ground-penetrating radar data and glaciological modeling},
	  author={Grab, Melchior and Mattea, Enrico and Bauder, Andreas and Huss, Matthias and Rabenstein, Lasse and Hodel, Elias and Linsbauer, Andreas and Langhammer, Lisbeth and Schmid, Lino and Church, Gregory and others},
	  journal={Journal of Glaciology},
	  volume={67},
	  number={266},
	  pages={1074--1092},
	  year={2021},
	  publisher={Cambridge University Press}
	}

	@article{linsbauer2021new,
	  title={The new Swiss Glacier Inventory SGI2016: From a topographical to a glaciological dataset},
	  author={Linsbauer, Andreas and Huss, Matthias and Hodel, Elias and Bauder, Andreas and Fischer, Mauro and Weidmann, Yvo and B{\"a}rtschi, Hans and Schmassmann, Emanuel},
	  journal={Frontiers in Earth Science},
	  pages={774},
	  year={2021},
	  publisher={Frontiers}
	}


