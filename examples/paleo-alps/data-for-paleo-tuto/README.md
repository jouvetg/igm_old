
# Overview

This set-up gives a simple set-up to run a paleo glacier model in the European Alps in paleo times with different catchements (lyon, ticino, rhine, linth glaciers) with IGM (the Instructed Glacier Model) e.g. around the last glacial maximum (LGM, 24 BP in the Alps).

# Inputs files

Input files are found in the folder data-for-paleo-tuto. There is:
 
    - Netcdf files that contains the present-day topography (SRTM data) after substracting present-day glaciers (dataset by Millan and al. 2022), and present-day lakes (Swisstopo data).
    - The EPICA climate temperature difference signal to drive the climate forcing (Ref: Jouzel, Jean; Masson-Delmotte, Valerie (2007): EPICA Dome C Ice Core 800KYr deuterium data and temperature estimates)
    - some flowlines usefull to plot result in plot-result.py

# Usage
	
Make sure the IGM's dependent libraries ar installed, or activate your igm environment with conda

		conda activate igm
	 
You may change the 'area', 'resolution', or the surface mass balance parametrization, and then run igm with 

		python igm-run.py
		
Don't forget to clean behind you:

		sh clean.sh

# Vizualize results

After any run, you may plot some results with companion python scripts (plot-result.py), or vizualize results with `ncview ex.nc`.
