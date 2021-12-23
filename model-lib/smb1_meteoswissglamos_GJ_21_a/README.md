
### <h1 align="center" id="title">SMB Emulator </h1>

# Overview   

This Surface Mass Balance model maps climate data to surface mass balance, it takes temperature (in °C) and precipitation (in m/y ice eq.) in inputs, and return the net surface mass balance (in m/y ice eq.). This convolutional neural network (CNN) was trained using climate data from meteoswiss (https://www.meteosuisse.admin.ch) from 1980 to 2013 and mass balance data from GLAMOS (https://www.glamos.ch/) on Aletsch Glacier. This must be understood as a preliminary attempt to include data-driven surface mass balance in IGM. I have not lead any analysis to assess the accuracy of the neural network. At least, it performs reasonably well on Aletsch Glacier (the training glacier) from 2017 to 2100 compared to the traditional accumulation/melt model.
