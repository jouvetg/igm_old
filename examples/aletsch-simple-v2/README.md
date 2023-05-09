
# Overview

This is the same example as aletsch-simple but modified with version 2 of the emulator that includes physics-informed deep learning emulator. This is done to help migrating from former emualtor ('v1') to new physics informed deel learning emaultors ('v2'). Make sure to be familiar with aletsch-simple before to run this example.

To migrate from version 'v1' to 'v2', you need to :

- Set glacier.config.version = 'v2'
- Set glacier.config.f21_pinnbp_GJ_23_a to a Version-2 emulator (e.g. f21_pinnbp_GJ_23_a), or set to empty string '' to train from scratch.
- Set the sliding coeeficient to a meaning full value, e.g. glacier.config.init_slidingco = 10000 , the unit has changed between v1 (km MPa-3 a-1) and v2 (m MPa-3 a-1), therefore makes sure to multiply by 1000 the v1 parameter.
- Optionally, you may play with the following parameters that control frequency of retraining and learning rating:

``
glacier.config.retrain_iceflow_emulator_freq = 5
glacier.config.retrain_iceflow_emulator_lr   = 0.00002
``

That's all!


