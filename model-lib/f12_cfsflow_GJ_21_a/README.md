### <h1 align="center" id="title">Ice Flow Emulator</h1>

# Model

Stokes-based glacier model CfsFlow (Jouvet and others, 2008)

# Training glaciers (dataset surflib3d_big_100)

10 synthetic glaciers using ice-free existing topographies (see the paper below)

# Emulated mapping

(thk,slopsurfx,slopsurfy,strflowctrl) -> (ubar,vbar,uvelsurf,vvelsurf)

# Resolutions

Native 100 m. Downscaled and upscaled to 200 m and 50 m.

# Reference

	@article{IGM2,
	  author       = "G. Jouvet",  
	  title        = "Inversion of a Stokes ice flow model emulated by deep learning",
	  journal      = "Journal of Glaciology",
	  year         = 2022,
	}
