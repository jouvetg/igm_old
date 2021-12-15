
### <h1 align="center" id="title">The Library of emulated models </h1>

You may find trained and ready-to-use ice flow emulators in the folder `T_M_I_Y_V/R/`, where T is the type of emulator defined by input and output lists, M is the name of the instructor model, I are the initials of the person who did the training, Y is the year the emulator was pusblished, and V is the version (in case several versions are produced per year), and R is the spatial resolution in meters. Information of the training dataset is provided in a dedicated README coming along the emulator. Here is a list of emulator type so far:
 
        self.mappings["f2"] = {
            "fieldin": ["thk", "slopsurfx", "slopsurfy"],
            "fieldout": ["ubar", "vbar"],
        }

        self.mappings["f10"] = {
            "fieldin": ["thk", "slopsurfx", "slopsurfy", "strflowctrl"],
            "fieldout": ["ubar", "vbar"],
        }

        self.mappings["f11"] = {
            "fieldin": ["thk", "slopsurfx", "slopsurfy", "arrhenius","slidingco"],
            "fieldout": ["ubar", "vbar"],
        }

        self.mappings["f12"] = {
            "fieldin": ["thk", "slopsurfx", "slopsurfy", "strflowctrl"],
            "fieldout": ["ubar", "vbar", "uvelsurf", "vvelsurf"],
        }

        self.mappings["f13"] = {
            "fieldin": ["thk", "slopsurfx", "slopsurfy", "arrhenius","slidingco"],
            "fieldout": ["ubar", "vbar", "uvelsurf", "vvelsurf"],
        }
        
        self.mappings["smb"] = {
            "fieldin": ["mask", "precipitation_0", "precipitation_1", "precipitation_2", "precipitation_3", "precipitation_4", "precipitation_5", "precipitation_6", "precipitation_7", "precipitation_8" , "precipitation_9", "precipitation_10", "precipitation_11", "air_temp_0", "air_temp_1", "air_temp_2", "air_temp_3", "air_temp_4", "air_temp_5", "air_temp_6", "air_temp_7", "air_temp_8", "air_temp_9", "air_temp_10", "air_temp_11"],            
            "fieldout": ["smb"],
        }

If you are unhappy with the proposed list of emulators, consider training your own with the [Deep Learning Emulator](https://github.com/jouvetg/dle).

