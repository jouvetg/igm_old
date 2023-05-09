# Overview

This is the same example as aletsch-invert but modified with version 2 of the emulator that includes physics-informed deep learning emulator. This is done to help migrating from former emualtor ('v1') to new physics informed deel learning emaultors ('v2'). Make sure to be familiar with aletsch-invert before to run this example.

To migrate from version 'v1' to 'v2', you need to :

- Set glacier.config.version = 'v2'
- the variable strflowctrl as disappeared, only optimize slidinco now.
- Set glacier.config.f21_pinnbp_GJ_23_a to a Version-2 emulator (e.g. f21_pinnbp_GJ_23_a), or set to empty string '' to train from scratch.
- Set the sliding coeeficient to a meaning full value, e.g. glacier.config.init_slidingco = 10000 , the unit has changed between v1 (km MPa-3 a-1) and v2 (m MPa-3 a-1), therefore makes sure to multiply by 1000 the v1 parameter.

WARNING: This is a very prelimanary version, a lot of tests / exploration of parameters would be needed. To be used with care. 

 
