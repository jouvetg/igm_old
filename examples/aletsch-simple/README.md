
# Overview

This set-up permits to simulate the evolution of the Great Aletsch Glacier (Switzerland) in two simple setting -- always starting from ice-free conditions. First, we force Equilibrium Line Altitudes (ELAs) in a simple mass balance model to simulate a 125-year-long advance followed by a 125-year-long retreat. Second, we build our own mass balance routine to simulate an oscilating ELA and produce an oscilating glacier advance/retreat behaviour. Our surface mass balance is a simple function of the elevation with 4 parameters, which are ELA, accumulation gradient, and ablation gradient, and maximum accumlation rate.

# Inputs files

Input files include geological inputs (geology.nc) and mass balance inputs for the first case (mb_simple_param.txt).

# Usage

First copy the igm code or export the PYTHONPATH, 

		cp ../../src/igm.py . # or export PYTHONPATH=../../src/
	
Make sure the IGM's dependent libraries ar installed, or activate your igm environment with conda

		conda activate igm
	 
The following command permits to run the first example:

		python -c "from igm import igm ; igm = igm() ; igm.run()" \
		       --tstart 1000 \
		       --tend 1250 \
		       --tsave 2 \
		       --cfl 0.3 \
		       --init_strflowctrl 90 \
		       --model_lib_path ../../model-lib/f12_cfsflow \
		       --type_mass_balance simple \
		       --usegpu True
		       
or equivalently

		python igm-run-simple.py

In the second example, we have defined our own mass balance routine in *igm-run-sinus.py* (just look at it), so it remains to run:

		python igm-run-sinus.py
	
Don't forget to clean behind you:

		sh clean.sh

# Vizualize results

After any run, you may vizualize results with `ncview ex.nc`, or plot png snapshots on the fly activating the `--plot_result True` option.

# Going further

You may look at other parameters typing (or looking at igm.py):

	python -c "from igm import igm ; igm = igm() " --help
	
e.g. activate the GPU with `--usegpu True`
