
# Overview

This set-up permits to run IGM with a synthetic glacier until steady steady. It also includes particle tracking.

# Usage

Make sure the IGM's dependent libraries ar installed, or activate your igm environment with conda

		conda activate igm
	 
Then you may run

		python make-syntetic-glacier.py

to create th input file geology.nc for IGM. Then you may run IGM with the following comand:

		python igm-run.py

