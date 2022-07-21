#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 07:07:44 2022

@author: gjouvet
"""

import os, sys, shutil, glob, csv, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from netCDF4 import Dataset
from scipy.interpolate import interp1d, RectBivariateSpline

x=np.arange(0,100)*100  # make x-axis, lenght 10 km, resolution 100 m
y=np.arange(0,200)*100  # make y-axis, lenght 20 km, resolution 100 m

X,Y=np.meshgrid(x,y)

topg = 1000 + 0.15*Y + ((X-5000)**2)/50000 # define the bedrock topography

# usurf = topg # define the top surface topography assuming ice thickness = zero

# plt.imshow(topg,origin='lower'); plt.colorbar()  # uncomemnt to vizualize

print('write_ncdf ----------------------------------')
    
nc = Dataset('geology.nc','w', format='NETCDF4') 
 
nc.createDimension('y',len(y))
yn = nc.createVariable('y',np.dtype('float64').char,('y',))
yn.units = 'm'
yn.long_name = 'y'
yn.standard_name = 'y'
yn.axis = 'Y'
yn[:] = y
 
nc.createDimension('x',len(x))
xn = nc.createVariable('x',np.dtype('float64').char,('x',))
xn.units = 'm'
xn.long_name = 'x'
xn.standard_name = 'x'
xn.axis = 'X'
xn[:] = x

E = nc.createVariable('topg', np.dtype("float32").char, ("y", "x"))
E.long_name = 'basal topography'
E.units     = 'm'
E.standard_name = 'topg'
E[:]  = topg
        
#E = nc.createVariable('usurf', np.dtype("float32").char, ("y", "x"))
#E.long_name = 'surface topography'
#E.units     = 'm'
#E.standard_name = 'usurf'
#E[:]  = usurf

nc.close()
