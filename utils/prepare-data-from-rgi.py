import oggm.cfg as cfg
from oggm import utils, workflow, tasks, graphics

# import xarray
import numpy as np
import matplotlib.pyplot as plt
import os, glob
from netCDF4 import Dataset
import argparse
from pyproj import Transformer
import scipy
import json

def str2bool(v):
    return v.lower() in ("true", "1")

parser = argparse.ArgumentParser(
    description="This script uses OGGM utilities and GlaThiDa dataset to prepare data \
                                 for the IGM model for a specific glacier given the RGI ID. \
                                 The user must provides RGI ID \
                                 (check GLIMS VIeWER : https://www.glims.org/maps/glims) \
                                 By default, data are already posprocessed, with spatial \
                                 resolutio of 100 and border of 30. \
                                 For custom spatial resolution, and the size of 'border' \
                                 to keep a safe distance to the glacier margin, one need\
                                 to set preprocess option to False \
                                 The script returns the geology.nc file as necessary for run \
                                 IGM for a forward glacier evolution run, and optionaly \
                                 observation.nc that permit to do a first step of data assimilation & inversion. \
                                 Data are currently based on COPERNIUS DEM 90 \
                                 the RGI, and the ice thckness and velocity from (MIllan, 2022) \
                                 For the ice thickness in geology.nc, the use can choose \
                                 between farinotti2019 or millan2022 dataset \
                                 Script written by G. Jouvet & F. Maussion"
)

parser.add_argument(
    "--RGI", type=str, default="RGI60-11.01450", help="RGI ID"
)  # malspina RGI60-01.13696
parser.add_argument(
    "--preprocess", type=str2bool, default=True, help="Use preprocessing"
)

parser.add_argument(
    "--dx",
    type=int,
    default=100,
    help="Spatial resolution (need preprocess false to change it)",
)
parser.add_argument(
    "--border",
    type=int,
    default=30,
    help="Safe border margin  (need preprocess false to change it)",
)

parser.add_argument(
    "--thk_source",
    type=str,
    default="farinotti2019",
    help="millan2022 or farinotti2019 in geology.nc",
)

parser.add_argument(
    "--geology", type=str2bool, default=True, help="Make geology file (for IGM forward)"
)
parser.add_argument(
    "--observation",
    type=str2bool,
    default=False,
    help="Make observation file (for IGM inverse)",
)

parser.add_argument(
    "--geology_file", type=str, default="geology.nc", help="Name of the geology file"
)
parser.add_argument(
    "--observation_file",
    type=str,
    default="observation.nc",
    help="Name of the observation file",
)

config = parser.parse_args()


def oggm_util_preprocess(RGIs, config, WD="OGGM-prepro"):

    print("oggm_util_preprocess ----------------------------------")

    # Initialize OGGM and set up the default run parameters
    cfg.initialize()

    cfg.PARAMS["continue_on_error"] = False
    cfg.PARAMS["use_multiprocessing"] = False

    # Where to store the data for the run - should be somewhere you have access to
    cfg.PATHS["working_dir"] = utils.gettempdir(dirname=WD, reset=True)

    # We need the outlines here
    rgi_ids = utils.get_rgi_glacier_entities(RGIs)

    base_url = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/exps/igm_v1"
    gdirs = workflow.init_glacier_directories(
        rgi_ids, prepro_border=30, from_prepro_level=2, prepro_base_url=base_url
    )

    path_ncdf = []
    for gdir in gdirs:
        path_ncdf.append(gdir.get_filepath("gridded_data"))

    return gdirs, path_ncdf


def oggm_util(RGIs, config, WD="OGGM-dir"):

    print("oggm_util ----------------------------------")

    # Initialize OGGM and set up the default run parameters
    cfg.initialize()

    cfg.PARAMS["continue_on_error"] = False
    cfg.PARAMS["use_multiprocessing"] = False

    # Map resolution parameters
    cfg.PARAMS["grid_dx_method"] = "fixed"
    cfg.PARAMS["fixed_dx"] = config.dx  # m spacing
    cfg.PARAMS[
        "border"
    ] = config.border  # can now be set to any value since we start from scratch
    cfg.PARAMS["map_proj"] = "utm"

    # Where to store the data for the run - should be somewhere you have access to
    cfg.PATHS["working_dir"] = utils.gettempdir(dirname=WD, reset=True)

    # We need the outlines here
    rgi_ids = utils.get_rgi_glacier_entities(RGIs)

    # Go - we start from scratch, i.e. we cant download from Bremen
    gdirs = workflow.init_glacier_directories(rgi_ids)

    # # gdirs is a list of glaciers. Let's pick one
    for gdir in gdirs:

        # https://oggm.org/tutorials/stable/notebooks/dem_sources.html
        tasks.define_glacier_region(gdir, source="DEM3")
        # Glacier masks and all
        tasks.simple_glacier_masks(gdir)

    # https://oggm.org/tutorials/master/notebooks/oggm_shop.html
    # If you want data we havent processed yet, you have to use OGGM shop
    from oggm.shop.millan22 import (
        thickness_to_gdir,
        velocity_to_gdir,
        compile_millan_statistics,
    )

    # This applies a task to a list of gdirs
    workflow.execute_entity_task(thickness_to_gdir, gdirs)
    workflow.execute_entity_task(velocity_to_gdir, gdirs)

    from oggm.shop import bedtopo

    workflow.execute_entity_task(bedtopo.add_consensus_thickness, gdirs)

    # We also have some diagnostics if you want
    df = compile_millan_statistics(gdirs)
    print(df.T)

    path_ncdf = []
    for gdir in gdirs:
        path_ncdf.append(gdir.get_filepath("gridded_data"))

    return gdirs, path_ncdf


######################################


def read_GlaThiDa(x, y, usurf, proj):

    print("read_GlaThiDa ----------------------------------")

    from scipy.interpolate import RectBivariateSpline
    import pandas as pd

    if not os.path.exists("glathida"):
        os.system("git clone https://gitlab.com/wgms/glathida")

    files = ["glathida/data/point.csv"]
    files += glob.glob("glathida/submissions/*/point.csv")

    transformer = Transformer.from_crs(proj, "epsg:4326", always_xy=True)

    lonmin, latmin = transformer.transform(min(x), min(y))
    lonmax, latmax = transformer.transform(max(x), max(y))

    transformer = Transformer.from_crs("epsg:4326", proj, always_xy=True)

    print(x.shape, y.shape, usurf.shape)

    fsurf = RectBivariateSpline(x, y, np.transpose(usurf))

    df = pd.concat(
        [pd.read_csv(file) for file in files],
        ignore_index=True
    )
    mask = (
        (lonmin <= df["lon"])
        & (df["lon"] <= lonmax)
        & (latmin <= df["lat"])
        & (df["lat"] <= latmax)
        & df["elevation"].notnull()
        & df["date"].notnull()
        & df["elevation_date"].notnull()
    )
    df = df[mask]

    # Filter by date gap in second step for speed
    mask = (
        df["date"].str.slice(0, 4).astype(int) -
        df["elevation_date"].str.slice(0, 4).astype(int)
    ).abs().le(1)
    df = df[mask]

    # Compute thickness relative to prescribed surface
    xx, yy = transformer.transform(df["lon"], df["lat"])
    bedrock = df["elevation"] - df["thickness"]
    elevation_normalized = fsurf(xx, yy, grid=False)
    thickness_normalized = np.maximum(elevation_normalized - bedrock, 0)

    # Rasterize thickness
    thickness_gridded = pd.DataFrame({
        'col': np.floor(xx - np.min(x) / (x[1] - x[0])).astype(int),
        'row': np.floor(yy - np.min(y) / (y[1] - y[0])).astype(int),
        'thickness': thickness_normalized
    }).groupby(['row', 'col'], as_index=False).mean()
    thkobs = np.full((y.shape[0], x.shape[0]), np.nan)
    thkobs[tuple(zip(*thickness_gridded.index))] = thickness_gridded
    return thkobs


########################################################

if config.preprocess:
    gdirs, paths_ncdf = oggm_util_preprocess([config.RGI], config)
else:
    gdirs, paths_ncdf = oggm_util([config.RGI], config)

print("read ncdf ----------------------------------")

nc = Dataset(paths_ncdf[0], "r+")
for var in nc.variables:
    vars()[var] = np.squeeze(nc.variables[var]).astype("float32")
    if not (var == "x") | (var == "y"):
        vars()[var] = np.flipud(vars()[var])
nc.close()
y = np.flip(y)

print("read projection info ----------------------------------")

fff = paths_ncdf[0].split("gridded_data.nc")[0] + "glacier_grid.json"
with open(fff, "r") as f:
    data = json.load(f)
proj = data["proj"]

########################################################

print("arrange data in and out the mask, and nan data -----")

for v in ["millan_ice_thickness", "millan_vx", "millan_vy"]:

    vars()[v] = np.where(np.isnan(vars()[v]), 0, vars()[v])
    vars()[v] = np.where(glacier_mask, vars()[v], 0)

for v in ["consensus_ice_thickness"]:

    vars()[v] = np.where(np.isnan(vars()[v]), 0, vars()[v])

######################################################

if config.observation:
    thkobs = read_GlaThiDa(x, y, topo, proj)
    thkobs = np.where(glacier_mask, thkobs, np.nan)

if config.thk_source == "millan2022":
    ice_thickness = millan_ice_thickness
#   ice_thickness = scipy.ndimage.uniform_filter(millan_ice_thickness,3)

elif config.thk_source == "farinotti2019":
    ice_thickness = consensus_ice_thickness

else:
    print("select right ice thk source")

######## make dictionary with qunattities description and unit s

var_info = {}
var_info["icemaskobs"] = ["Accumulation Mask", "bool"]
var_info["usurfobs"] = ["Surface Topography", "m"]
var_info["thkobs"] = ["Ice Thickness", "m"]
var_info["thkinit"] = ["Ice Thickness", "m"]
var_info["uvelsurfobs"] = ["x surface velocity of ice", "m/y"]
var_info["vvelsurfobs"] = ["y surface velocity of ice", "m/y"]
var_info["thk"] = ["Ice Thickness", "m"]
var_info["usurf"] = ["Surface Topography", "m"]
var_info["icemask"] = ["Ice mask", "no unit"]

#########################################################

if config.geology:

    print("Write geology file -----")

    nc = Dataset(config.geology_file, "w", format="NETCDF4")

    nc.createDimension("y", len(y))
    yn = nc.createVariable("y", np.dtype("float64").char, ("y",))
    yn.units = "m"
    yn.long_name = "y"
    yn.standard_name = "y"
    yn.axis = "Y"
    yn[:] = y

    nc.createDimension("x", len(x))
    xn = nc.createVariable("x", np.dtype("float64").char, ("x",))
    xn.units = "m"
    xn.long_name = "x"
    xn.standard_name = "x"
    xn.axis = "X"
    xn[:] = x

    vars_to_save = {}

    vars_to_save["usurf"] = "topo"
    vars_to_save["thk"] = "ice_thickness"

    for nigm in vars_to_save:

        noggm = vars_to_save[nigm]

        E = nc.createVariable(nigm, np.dtype("float32").char, ("y", "x"))
        E.long_name = var_info[nigm][0]
        E.units = var_info[nigm][1]
        E.standard_name = nigm
        print(noggm, vars()[noggm].shape)
        E[:] = vars()[noggm]

    nc.close()

#########################################################

if config.observation:

    print("Write observation file -----")

    nc = Dataset(config.observation_file, "w", format="NETCDF4")

    nc.createDimension("y", len(y))
    yn = nc.createVariable("y", np.dtype("float64").char, ("y",))
    yn.units = "m"
    yn.long_name = "y"
    yn.standard_name = "y"
    yn.axis = "Y"
    yn[:] = y

    nc.createDimension("x", len(x))
    xn = nc.createVariable("x", np.dtype("float64").char, ("x",))
    xn.units = "m"
    xn.long_name = "x"
    xn.standard_name = "x"
    xn.axis = "X"
    xn[:] = x

    vars_to_save = {}

    vars_to_save["usurfobs"] = "topo"
    vars_to_save["thkobs"] = "thkobs"
    vars_to_save["icemaskobs"] = "glacier_mask"
    vars_to_save["uvelsurfobs"] = "millan_vx"
    vars_to_save["vvelsurfobs"] = "millan_vy"
    vars_to_save["thkinit"] = "millan_ice_thickness"

    for nigm in vars_to_save:

        noggm = vars_to_save[nigm]

        E = nc.createVariable(nigm, np.dtype("float32").char, ("y", "x"))
        E.long_name = var_info[nigm][0]
        E.units = var_info[nigm][1]
        E.standard_name = nigm
        print(noggm, vars()[noggm].shape)
        E[:] = vars()[noggm]

    nc.close()
