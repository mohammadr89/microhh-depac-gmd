import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import netCDF4 as nc

import microhh_tools as mht
import helpers as hlp
from constants import *
from lsm_input import LSM_input

"""
Settings.
"""
float_type = "f8"

xsize = 7680
ysize = 3840
zsize = 3840

itot = 768
jtot = 384
ktot = 128

start_hour = 0
end_hour = 24

sunrise_hour = 6.0
sunset_hour = 18.0

base_date = "20-09-20"
hour = int(start_hour)
minute = 0
second = 0
datetime_str = f"{base_date} {hour:02d}:{minute:02d}:{second:02d}"

lat=51.97
lon=4.926

dx = xsize/itot  # Grid spacing in x
dy = ysize/jtot  # Grid spacing in y
print(f"\nGrid spacing:")
print(f"dx = {dx}m")
print(f"dy = {dy}m")

sw_plume_rise = False
sw_chemistry = True
sw_land_surface = True

# LAND SURFACE CONFIGURATION
# Choose one of: "grass", "forest", "heterogeneous"
land_surface_type = "forest"  # Change this to "grass", "forest", or "heterogeneous"

# Surface type definitions.
# DEPAC lu codes (from depac_lu.inc): 1=grass, 4=coniferous_forest, 5=deciduous_forest

surface_type_props = {
    "grassland": {
        "lu":         1,   
        "lai":        3.1,
        "z0m":        0.03,
        "z0h":        0.003,
        "c_veg":      1.0,
        "gD":         0.0,
        "rs_veg_min": 100,
        "root_frac":  [0.24, 0.41, 0.31, 0.04],
    },
    "deciduous_forest": {
        "lu":         5,      # ilu_deciduous_forest
        "lai":        4.0,   
        "z0m":        0.75,
        "z0h":        0.075, 
        "c_veg":      1.0,
        #  MicroHH uses Pa:
        "gD": 0.0003,   # = 0.03 hPa⁻¹ converted to Pa⁻¹
        "rs_veg_min": 175,
        "root_frac":  [0.24, 0.38, 0.29, 0.07],
    },
    # "coniferous_forest": {
    #     "lu":         4,      # ilu_coniferous_forest
    #     "lai":        5.0,
    #     "z0m":        0.75,
    #     "z0h":        0.075,
    #     "c_veg":      1.0,
    #     "gD":         0.0003,
    #    "rs_veg_min": 175,
    #    "root_frac":  [0.24, 0.38, 0.29, 0.07],
    # },
}

ini = mht.Read_namelist("plume_chem.ini.base")

if sw_chemistry:

    # Read TUV output table.
    columns = [
        "time",
        "sza",
        "jo31d",
        "jh2o2",
        "jno2",
        "jno3",
        "jn2o5",
        "jch2or",
        "jch2om",
        "jch3o2h",
    ]
    tuv = pd.read_table(
        "plume_chem_tuv_output.txt",
        sep="\\s+",
        skiprows=12,
        skipfooter=1,
        engine="python",
        names=columns,
        index_col="time",
    )

    tuv = tuv.loc[start_hour:end_hour]
    tuv.index *= 3600
    tuv.index -= tuv.index.values[0]

    emi_no = np.zeros(tuv.index.size)
    emi_isop = np.zeros(tuv.index.size)

    # species = {"nh3": 0.0} 
    species = {"nh3": 7.06167e-9} # Equivalent to 5 ug/m3 in T=20 C and P=101325 Pa

    deposition_species = ["nh3"]
else:
    species = {}


def stretched_vertical_grid(z_bottom, z_top, n_layers, stretch_factor):
    # Create a normalized grid in [0, 1]
    eta = np.linspace(0, 1, n_layers + 1)  # n_layers + 1 points (including top and bottom)
    stretched_eta = (np.exp(stretch_factor * eta) - 1) / (np.exp(stretch_factor) - 1)
    zh = z_bottom + (z_top - z_bottom) * stretched_eta
    z = 0.5 * (zh[:-1] + zh[1:])
    return z, zh

z_bottom = 0
z_top = zsize
n_layers = ktot
stretch_factor = 1.6
z, zh = stretched_vertical_grid(z_bottom, z_top, n_layers, stretch_factor)

dz = np.diff(zh)

# dz = zsize / ktot
# z = np.arange(0.5 * dz, zsize, dz)


def profile(zi, v_bulk, dv, gamma_v, clip_at_zero=False):
    """
    Create well mixed profile with jump and constant lapse rate above.
    """
    k_zi = np.abs(z - zi).argmin()
    profile = np.zeros(ktot)
    profile[:k_zi] = v_bulk
    profile[k_zi:] = v_bulk + dv + gamma_v * (z[k_zi:] - zi)

    if clip_at_zero:
        profile[profile < 0] = 0.0

    return profile


thl = profile(zi=600, v_bulk=291,   dv=2,     gamma_v=0.006)
qt  = profile(zi=600, v_bulk=7.5e-3, dv=-0.3e-3, gamma_v=-0.002e-3, clip_at_zero=True)
u = np.ones(ktot) * 5
nudgefac = np.ones(ktot) / 10800

time = np.linspace(start_hour*3600, end_hour*3600, 97)
hour_of_day = (time / 3600) % 24
daytime_mask = (hour_of_day >= sunrise_hour) & (hour_of_day <= sunset_hour)

wthl = 0.2 * np.sin((2.0 * np.pi /24) * (hour_of_day - 6.0))
wqt = 8e-5 * np.maximum(0 , np.sin((2.0 * np.pi /24) * (hour_of_day - 6.0)))
max_rad = 650

sw_flux_dn = np.zeros_like(time)
sw_flux_dn[daytime_mask] = max_rad * np.sin(np.pi * (hour_of_day[daytime_mask] - sunrise_hour) / (sunset_hour - sunrise_hour))
sw_flux_dn[sw_flux_dn < 0] = 0
sw_flux_up = 0.2 * sw_flux_dn

lw_flux_dn = np.ones_like(sw_flux_dn) * 334
lw_flux_up = np.ones_like(sw_flux_dn) * 472

def add_nc_var(name, dims, nc, data):
    if dims is None:
        var = nc.createVariable(name, np.float64)
    else:
        var = nc.createVariable(name, np.float64, dims)
    var[:] = data

nc_file = nc.Dataset("plume_chem_input.nc", mode="w", datamodel="NETCDF4", clobber=True)

add_nc_var("start_hour", None, nc_file, start_hour)
add_nc_var("t_sunrise", None, nc_file, sunrise_hour)
add_nc_var("t_sunset", None, nc_file, sunset_hour)

nc_file.createDimension("z", ktot)
add_nc_var("z", ("z"), nc_file, z)

# Atmospheric input.
nc_group_init = nc_file.createGroup("init")

add_nc_var("u", ("z"), nc_group_init, u)
add_nc_var("thl", ("z"), nc_group_init, thl)
add_nc_var("qt", ("z"), nc_group_init, qt)
add_nc_var("nudgefac", ("z"), nc_group_init, nudgefac)
add_nc_var("qt_nudge", ("z"), nc_group_init, qt)
nc_tdep = nc_file.createGroup("timedep");
nc_tdep.createDimension("time_surface", time.size)
add_nc_var("time_surface", ("time_surface"), nc_tdep, time-time[0])
add_nc_var("thl_sbot", ("time_surface"), nc_tdep, wthl)
add_nc_var("qt_sbot", ("time_surface"), nc_tdep, wqt)

if (sw_chemistry):
    nc_chem = nc_file.createGroup("timedep_chem");
    nc_chem.createDimension("time_chem", tuv.index.size)

    add_nc_var("time_chem", ("time_chem"), nc_chem, tuv.index)
    add_nc_var("jo31d", ("time_chem"), nc_chem, tuv.jo31d)
    add_nc_var("jh2o2", ("time_chem"), nc_chem, tuv.jh2o2)
    add_nc_var("jno2", ("time_chem"), nc_chem, tuv.jno2)
    add_nc_var("jno3", ("time_chem"), nc_chem, tuv.jno3)
    add_nc_var("jn2o5", ("time_chem"), nc_chem, tuv.jn2o5)
    add_nc_var("jch2or", ("time_chem"), nc_chem, tuv.jch2or)
    add_nc_var("jch2om", ("time_chem"), nc_chem, tuv.jch2om)
    add_nc_var("jch3o2h", ("time_chem"), nc_chem, tuv.jch3o2h)
    add_nc_var("emi_isop", ("time_chem"), nc_chem, emi_isop)
    add_nc_var("emi_no", ("time_chem"), nc_chem, emi_no)

    for name, value in species.items():
        profile = np.ones(ktot, dtype=np.float64)*value
        add_nc_var(name, ("z"), nc_group_init, profile)
        add_nc_var("{}_inflow".format(name), ("z"), nc_group_init, profile)

if (sw_land_surface):
    nc_soil = nc_file.createGroup("soil")
    nc_soil.createDimension("z", 4)
    add_nc_var("z", ("z"), nc_soil, np.array([-1.945, -0.64, -0.175, -0.035]))
    add_nc_var("theta_soil", ("z"), nc_soil, np.array([0.34, 0.30, 0.27, 0.24]))
    add_nc_var("t_soil", ("z"), nc_soil, np.array([282, 287, 290, 286]))
    add_nc_var("index_soil", ("z"), nc_soil, np.ones(4) * 2)

    add_nc_var("sw_flux_dn", ("time_surface"), nc_tdep, sw_flux_dn)
    add_nc_var("sw_flux_up", ("time_surface"), nc_tdep, sw_flux_up)
    add_nc_var("lw_flux_dn", ("time_surface"), nc_tdep, lw_flux_dn)
    add_nc_var("lw_flux_up", ("time_surface"), nc_tdep, lw_flux_up)


x0 = 400
y0 = ysize/2.
z0 = 0

sigma_x = (xsize/itot)*0.5
sigma_y = (ysize/jtot)*0.5
sigma_z = (xsize/itot)*0.5

strength_nh3  = (0.01e-3) / 17.031  # 0.01 g/s (Livestock farm)

emi = hlp.Emissions()

y_offsets = [-28, -9, 9, 18]  # Specified y-offsets

for y_offset in y_offsets:
    x = x0
    y = y0 + y_offset
    z = z0

    # Only adding NH3 as the source as requested
    if (sw_chemistry):
        emi.add('nh3', strength_nh3, True, x, y, z, sigma_x, sigma_y, sigma_z)

if (sw_land_surface):
    blocksize_i = 1
    blocksize_j = 1
    block_points = blocksize_i * blocksize_j
    region_sizex = itot // blocksize_i
    region_sizey = jtot // blocksize_j

    print(f"\nBlock structure:")
    print(f"Each block is {blocksize_i}×{blocksize_j} grid points")
    print(f"Maximum possible regions: {region_sizex} (x) × {region_sizey} (y)")
    print(f"Selected land surface type: {land_surface_type}")

    if land_surface_type == "heterogeneous":
        grass_upwind_end = int((1400/dx) // blocksize_i) * blocksize_i
        forest_end = int((1400/dx + 5000/dx) // blocksize_i) * blocksize_i
    elif land_surface_type == "grass":
        grass_upwind_end = itot  # All grass_upwind
        forest_end = itot        # No forest
    elif land_surface_type == "forest":
        grass_upwind_end = 0     # No grass_upwind
        forest_end = itot        # All forest
    else:
        raise ValueError(f"Unknown land_surface_type: {land_surface_type}. Use 'grass', 'forest', or 'heterogeneous'")

    grass_name  = next((k for k in surface_type_props if "grass"  in k), "grass")
    forest_name = next((k for k in surface_type_props if "forest" in k), "forest")

    print(f"\nDomain setup (aligned to block boundaries):")
    print(f"{grass_name}_upwind region: 0 to {grass_upwind_end*dx}m (0 to {grass_upwind_end} points)")
    print(f"{forest_name} region: {grass_upwind_end*dx}m to {forest_end*dx}m ({grass_upwind_end} to {forest_end} points)")
    print(f"{grass_name}_downwind region: {forest_end*dx}m to {itot*dx}m ({forest_end} to {itot} points)")

    if land_surface_type == "grass":
        grass_name = [k for k in surface_type_props if "grass" in k][0]
        mask_file = np.ones((jtot, itot), dtype=float_type)
        mask_file.tofile(f"{grass_name}.0000000")
        mask_list = grass_name
        print(f"Created: {grass_name}.0000000 (entire domain)")

    elif land_surface_type == "forest":
        forest_name = [k for k in surface_type_props if "forest" in k][0]
        mask_file = np.ones((jtot, itot), dtype=float_type)
        mask_file.tofile(f"{forest_name}.0000000")
        mask_list = forest_name
        print(f"Created: {forest_name}.0000000 (entire domain)")

    elif land_surface_type == "heterogeneous":
        grass_name  = [k for k in surface_type_props if "grass"  in k][0]
        forest_name = [k for k in surface_type_props if "forest" in k][0]

        mask_grass_upwind   = np.zeros((jtot, itot), dtype=float_type)
        mask_forest_file    = np.zeros((jtot, itot), dtype=float_type)
        mask_grass_downwind = np.zeros((jtot, itot), dtype=float_type)

        for j in range(0, jtot, blocksize_j):
            j_slice = slice(j, j + blocksize_j)
            for i in range(0, grass_upwind_end, blocksize_i):
                mask_grass_upwind[j_slice, slice(i, i+blocksize_i)] = 1
            for i in range(grass_upwind_end, forest_end, blocksize_i):
                mask_forest_file[j_slice, slice(i, i+blocksize_i)] = 1
            for i in range(forest_end, itot, blocksize_i):
                mask_grass_downwind[j_slice, slice(i, i+blocksize_i)] = 1

        mask_grass_upwind  .tofile(f"{grass_name}_upwind.0000000")
        mask_forest_file   .tofile(f"{forest_name}.0000000")
        mask_grass_downwind.tofile(f"{grass_name}_downwind.0000000")

        mask_list = f"{grass_name}_upwind,{forest_name},{grass_name}_downwind"
        print(f"Created: {grass_name}_upwind.0000000, {forest_name}.0000000, {grass_name}_downwind.0000000")

    # Initialize LSM
    # ls = LSM_input(itot, jtot, 4, sw_water=True, TF=float_type, debug=True, exclude_fields=['z0m', 'z0h'])
    ls = LSM_input(itot, jtot, 4, sw_water=True, TF=float_type, debug=True)

    if land_surface_type == "grass":
        mask_deciduous_forest = np.zeros((jtot, itot), dtype=bool)
        mask_grassland        = np.ones ((jtot, itot), dtype=bool)
        print("LSM setup: Pure grass domain")

    elif land_surface_type == "forest":
        mask_deciduous_forest = np.ones ((jtot, itot), dtype=bool)
        mask_grassland        = np.zeros((jtot, itot), dtype=bool)
        print("LSM setup: Pure deciduous forest domain")

    else:  # heterogeneous
        mask_deciduous_forest = np.zeros((jtot, itot), dtype=bool)
        mask_deciduous_forest[:, grass_upwind_end:forest_end] = True
        mask_grassland        = ~mask_deciduous_forest
        print("LSM setup: Heterogeneous domain (grass-forest-grass)")

    # Map each surface type name to its spatial mask.
    # To add a new surface type: add one line here.
    # mask_coniferous_forest = ...   # uncomment and define when needed
    surface_type_masks = {
        "grassland":        mask_grassland,
        "deciduous_forest": mask_deciduous_forest,
        # "coniferous_forest": mask_coniferous_forest,  # uncomment when needed
    }

    lu_map         = np.zeros((jtot, itot), dtype=float_type)
    lai_map        = np.zeros((jtot, itot), dtype=float_type)
    z0m            = np.zeros((jtot, itot), dtype=float_type)
    z0h            = np.zeros((jtot, itot), dtype=float_type)
    c_veg_map      = np.zeros((jtot, itot), dtype=float_type)
    gD_map         = np.zeros((jtot, itot), dtype=float_type)
    rs_veg_min_map = np.zeros((jtot, itot), dtype=float_type)
    root_frac      = np.zeros((4, jtot, itot), dtype=float_type)

    for name, props in surface_type_props.items():
        mask = surface_type_masks[name]
        lu_map        [mask] = props["lu"]
        lai_map       [mask] = props["lai"]
        z0m           [mask] = props["z0m"]
        z0h           [mask] = props["z0h"]
        c_veg_map     [mask] = props["c_veg"]
        gD_map        [mask] = props["gD"]
        rs_veg_min_map[mask] = props["rs_veg_min"]
        for layer in range(4):
            root_frac[layer][mask] = props["root_frac"][layer]

    # Apply to LSM
    ls["z0m"]       [:,:] = z0m
    ls["z0h"]       [:,:] = z0h
    ls["lai"]       [:,:] = lai_map
    ls["c_veg"]     [:,:] = c_veg_map
    ls["gD"]        [:,:] = gD_map
    ls["rs_veg_min"][:,:] = rs_veg_min_map
    ls["root_frac"] [:,:,:] = root_frac
    ls["water_mask"][:,:] = 0

    z0m.tofile("z0m.0000000")
    z0h.tofile("z0h.0000000")
    lu_map.tofile("lu_map.0000000")
    ls["rs_soil_min"]    [:,:] = 50
    ls["lambda_stable"]  [:,:] = 10
    ls["lambda_unstable"][:,:] = 10
    ls["cs_veg"]         [:,:] = 0.0
    ls["t_bot_water"]    [:,:] = 295
    ls["t_soil"]    [:,:,:] = 290
    ls["theta_soil"][:,:,:] = 0.30
    ls["index_soil"][:,:,:] = 2
    ls.check()
    ls.save_binaries(allow_overwrite=True)
    ls.save_netcdf("lsm_input.nc", allow_overwrite=True)

    ###########################
    #Add settings to .ini file.
    ###########################
    ini["grid"]["itot"] = itot
    ini["grid"]["jtot"] = jtot
    ini["grid"]["ktot"] = ktot
    ini["grid"]["xsize"] = xsize
    ini["grid"]["ysize"] = ysize
    ini["grid"]["zsize"] = zsize
    ini["stats"]["xymasklist"] = mask_list

    scalars = list(species.keys())
    ini["advec"]["fluxlimit_list"] = scalars
    ini["limiter"]["limitlist"] = scalars
    ini["fields"]["slist"] = scalars
    ini["boundary"]["scalar_outflow"] = scalars

    ini['grid']['lat'] = lat
    ini['grid']['lon'] = lon
    ini['deposition']['start_hour'] = start_hour
    ini['radiation']['max_rad'] =  max_rad
    ini["time"]["datetime_utc"] = datetime_str
    ini["time"]["endtime"] = (end_hour - start_hour) * 3600

    ini["source"]["sourcelist"] = emi.source_list
    ini["chemistry"]["swchemistry"] = sw_chemistry

    crosslist = ["u", "v", "w", "thl_fluxbot", "qt_fluxbot", "flux_inst",
             "ra", "rb", "obuk", "ustar", "ccomp_tot", "cw", "cstom",
             "csoil_eff", "cw_out", "cstom_out", "csoil_out", "rc_tot", "rc_eff",
             "rh_surface", "T_surface", "flux_nh3", "total_flux_mol_ha", "cstar1",
             "cstar2", "c_target", "c_diff", "c_extrap_diff","rho_target","T_target"]

    if (sw_chemistry):
        crosslist += list(species.keys())
        crosslist += [f"{x}_path" for x in species.keys()]
        crosslist += [f"vd{x}" for x in deposition_species]

    if (sw_land_surface):
        ini["boundary"]["swboundary"] = "surface_lsm"
        ini["boundary"]["sbcbot"] = "flux"
        ini["boundary"]["sbot"] = "0"
        ini["boundary"]["thl"] = "dirichlet"
        ini["boundary"]["qt"] = "dirichlet"
        ini["boundary"]["swtimedep"] = False
        ini["boundary"]["timedeplist"] = "empty"

        ini["radiation"]["swradiation"] = "prescribed"

    else:
        ini["boundary"]["swboundary"] = "surface"
        ini["boundary"]["sbcbot"] = "flux"
        ini["boundary"]["swtimedep"] = True
        ini["boundary"]["timedeplist"] = ["thl_sbot", "qt_sbot"]
        ini["radiation"]["swradiation"] = False


    if (sw_chemistry and sw_land_surface):
        ini["deposition"]["swdeposition"] = True
    else:
        ini["deposition"]["swdeposition"] = False

    ini["cross"]["crosslist"] = crosslist

    ini["source"]["source_x0"] = emi.x0
    ini["source"]["source_y0"] = emi.y0
    ini["source"]["source_z0"] = emi.z0
    ini["source"]["sigma_x"] = emi.sigma_x
    ini["source"]["sigma_y"] = emi.sigma_y
    ini["source"]["sigma_z"] = emi.sigma_z
    ini["source"]["strength"] = emi.strength
    ini["source"]["swvmr"] = emi.sw_vmr
    ini["source"]["line_x"] = emi.line_x
    ini["source"]["line_y"] = emi.line_y
    ini["source"]["line_z"] = emi.line_z

    nc_lsm = nc_file.createGroup("lsm_params")
    for name, props in surface_type_props.items():
        add_nc_var(f"lai_{name}", None, nc_lsm, props["lai"])
        add_nc_var(f"lu_{name}",  None, nc_lsm, props["lu"])

    nc_file.close() 

    ini.save("plume_chem.ini", allow_overwrite=True)
    print(f"First cell height: {dz[0]:.2f}m")
    print(f"Last cell height: {dz[-1]:.2f}m")
