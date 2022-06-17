
# Overview

This set-up permits to simulate the evolution of the Great Aletsch Glacier (Switzerland) in two simple setting -- always starting from ice-free conditions. First, we force Equilibrium Line Altitudes (ELAs) in a simple mass balance model to simulate a 125-year-long advance followed by a 125-year-long retreat. Second, we build our own mass balance routine to simulate an oscilating ELA and produce an oscilating glacier advance/retreat behaviour. Our surface mass balance is a simple function of the elevation with 4 parameters, which are ELA, accumulation gradient, and ablation gradient, and maximum accumlation rate.

# Inputs files

Input files include geological inputs (geology.nc) and mass balance inputs for the first case (mb_simple_param.txt).

# Usage 
	
Make sure the IGM's dependent libraries ar installed, or activate your igm environment with conda

		conda activate igm
	 
In the second example, we run *igm-run-simple.py* (just look at it):

		python igm-run-simple.py
		
You may change parameters directly in igm-run-simple.py or externally as follows:

		python igm-run-simple.py --usegpu True --plot_result True
		
You may look at other parameters typing:

		python igm-run-simple.py --help

In the second example, we have defined our own mass balance routine in *igm-run-sinus.py* (just look at it), so it remains to run:

		python igm-run-sinus.py
	
Don't forget to clean behind you:

		sh clean.sh

# Vizualize results

After any run, you may vizualize results with `ncview ex.nc`, or plot png snapshots on the fly activating the `--plot_result True` option.

