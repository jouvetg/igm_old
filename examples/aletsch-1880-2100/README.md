
# Overview

This set-up aims to closely reproduce the [simulations](https://www.geo.uzh.ch/~gjouvet/the-aletsch-glacier-module/) of the Great Aletsch Glacier (Switzerland) in the [past](https://www.cambridge.org/core/journals/journal-of-glaciology/article/modelling-the-retreat-of-grosser-aletschgletscher-switzerland-in-a-changing-climate/C877413079F73C5FC6131FC7BC031B69) and in the [future](https://www.cambridge.org/core/journals/journal-of-glaciology/article/future-retreat-of-great-aletsch-glacier/EB46DC696E0AB9528168F42595EE23D9) based on the CH2018 climate scenarios described in (Jouvet and al., JOG, 2011) and (Jouvet and Huss, JOG, 2019), respectively.

The goal of this set-up is to give a template of climate-based mass balance implementation within the TensorFlow framework, which can serve for future improvements of the Aletsch case, or adaptations to other mountain glaciers. This simulation relies on an accumulation and distributed temperature-index melt mass balance model (Hock, JOG, 1999). The closest and most recent description of the implemented model is given in (Jouvet and al., TC, 2020). This simulation does not stricly match the original implementation due to discrepencies in input data and ice flow models (emulated Stokes vs original Stokes). Therefore, melt-factors were re-calibrated to get the best match between observed and modelled top ice surface in the past.

# Inputs files

Input files include geological inputs (geology.nc), spatially-varying fields for the computation of melt factors and snow reditribution (massbalance.nc), as well as temperature and precipitation time serie data (temp_prec.dat). The data also provide some observed top surface ice topographies (in 1880, 1926, 1957, 1980, 1999, 2009, 2017 and 2016).

# Usage

First copy the igm code or export the PYTHONPATH, 

		cp ../../src/igm.py . # or export PYTHONPATH=../../src/
	
Make sure the IGM's dependent libraries ar installed, or activate your igm environment with conda

		conda activate igm
	 
Then you may run

		python igm-run.py

and try to change the starting year within igm-run.py to 1880, 1926, 1957, 1980, 1999, 2009, or 2017.

Last but not least, you may try (starting from 2017) to activate in igm-run.py a mass balance model, which is a convolutional neural network (CNN) trained from climate and mass balance data (from Aletsch and Rhone Glaciers). This can be seen as an alternative to the accumulation / melt model. Therefore, you must set igm.config.type_mass_balance to 'nn', provide a valid path for the mass balance emulator igm.config.smb_model_lib_path, and also force monthly climate inputs (instead of daily) setting igm.config.clim_time_resolution to 12.

Note that igm-run.py builds on the core igm class, and two class extensions (igm_smb_accmelt.py and igm_clim_aletsch.py), which implement the climate forcing and the mass balance, respectively.

# References	

	@article{hock1999distributed,
	  title={A distributed temperature-index ice-and snowmelt model including potential direct solar radiation},
	  author={Hock, Regine},
	  journal={Journal of glaciology},
	  volume={45},
	  number={149},
	  pages={101--111},
	  year={1999},
	  publisher={Cambridge University Press}
	}
	
	@article{jouvet2011modelling,
	  title={Modelling the retreat of Grosser Aletschgletscher, Switzerland, in a changing climate},
	  author={Jouvet, Guillaume and Huss, Matthias and Funk, Martin and Blatter, Heinz},
	  journal={Journal of Glaciology},
	  volume={57},
	  number={206},
	  pages={1033--1045},
	  year={2011},
	  publisher={Cambridge University Press}
	}
 
	@article{jouvet2019,
	  title={Future retreat of great Aletsch glacier},
	  author={Jouvet, Guillaume and Huss, Matthias},
	  journal={Journal of Glaciology},
	  volume={65},
	  number={253},
	  pages={869--872},
	  year={2019},
	  publisher={Cambridge University Press}
	}

	@article{jouvet2020,
	  title={Mapping the age of ice of Gauligletscher combining surface radionuclide contamination and ice flow modeling},
	  author={Jouvet, Guillaume and R{\"o}llin, Stefan and Sahli, Hans and Corcho, Jos{\'e} and Gn{\"a}gi, Lars and Compagno, Loris and Sidler, Dominik and Schwikowski, Margit and Bauder, Andreas and Funk, Martin},
	  journal={The Cryosphere},
	  volume={14},
	  number={11},
	  pages={4233--4251},
	  year={2020},
	  publisher={Copernicus GmbH}
	}

