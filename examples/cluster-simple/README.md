
# Overview

This set-up permits to simulate the evolution of a cluster of glaciers located around the Great Aletsch Glacier (Switzerland) using a simple (unreaslistic) mass balance model. The goals of this experiment are to i) to demonstrate the capability of IGM to simulate a network of glaciers (without any explicit individual identification) ii) to show that IGM behaves as if it has natural boundary conditions (allowing glaciers to outflow the computational domain).

# Inputs files

Input files include geological inputs (geology.nc) and mass balance inputs for the first case (mb_simple_param.txt).

# Usage

First copy the igm code or export the PYTHONPATH, 

		cp ../../src/igm.py . # or export PYTHONPATH=../../src/
	
Make sure the IGM's dependent libraries ar installed, or activate your igm environment with conda

		conda activate igm
	 
Run the command:

		python igm-run.py
	
Don't forget to clean behind you:

		sh clean.sh
		

