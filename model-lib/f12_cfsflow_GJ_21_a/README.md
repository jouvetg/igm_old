
### <h1 align="center" id="title">Ice Flow Emulator trained with CfsFlow</h1>

# Overview   

To train our ice flow emulator, we performed an ensemble of simulations to generate large datasets using a Stokes-based glacier model CfsFlow (Jouvet and others, 2008). The goal was to construct diverse states to obtain a heterogeneous dataset that a large variety of possible glaciers (large/narrow, thin/thick, flat/steep, long/small, fast/slow, straight/curved glaciers, . . . ) that can be met in future modelling. 

We simulate 200 years time evolution of 10 glaciers that are artificially built on existing ice-free topographies from the European Alps and New Zealand. For each valley, we ran CfsFlow at 100 m horizontal resolution and force equilibrium line altitudes in a simple mass balance model to simulate a 100-year-long advance followed by a 100-year-long retreat. The results were recorded every 2 years to
provide a wide range of dynamical states roughly representative of real world temperate glacier behaviour consisting of about 41 × 100
snapshots. Each simulation were repeated for an ensemble of 9 "ice flow strength parameter" (\tilde{A}) to cover a large panel of ice flow regimes from slow-shearing to fast-sliding. 

This emulator performs the following mapping: 
	(thk,slopsurfx,slopsurfy,strflowctrl) -> (ubar,vbar,uvelsurf,vvelsurf)

The emulator accuracy was tested using an independent dataset, and fidelity levels close to 90% were found with respect to solutions produced ith Cfsflow providing the test solution to be in the hull of the training data set.

The native resolution of the emulator is 100 m. A 200 m emulator obtained by data averaging is aslo available.

# Reference

	@article{IGM,
	  author       = "G. Jouvet, G. Cordonnier, B. Kim, M. Luethi, A. Vieli, A. Aschwanden",  
	  title        = "Deep learning speeds up ice flow modelling by several orders of magnitude",
	  journal      = "Journal of Glaciology",
	  year         = 2021,
	}


