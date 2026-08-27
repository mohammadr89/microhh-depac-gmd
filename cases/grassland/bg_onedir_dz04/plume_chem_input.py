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
ktot = 384

start_hour = 0
end_hour = 24

# Define sunrise and sunset hours (single source of truth)
sunrise_hour = 6.0
sunset_hour = 18.0

base_date = "20-09-20"
hour = int(start_hour)
minute = 0
second = 0
datetime_str = f"{base_date} {hour:02d}:{minute:02d}:{second:02d}"

lat=51.97
lon=4.926

# Define grid spacing explicitly
dx = xsize/itot  # Grid spacing in x
dy = ysize/jtot  # Grid spacing in y
print(f"\nGrid spacing:")
print(f"dx = {dx}m")
print(f"dy = {dy}m")

# Enable resolved plume rise:
sw_plume_rise = False

# Enable non-linear KPP chemistry:
sw_chemistry = True

# Enable land-surface model and more detailled deposition.
sw_land_surface = True

# LAND SURFACE CONFIGURATION
# Choose one of: "grass", "forest", "heterogeneous"
# land_surface_type = "heterogeneous"  # Change this to "grass", "forest", or "heterogeneous"
land_surface_type = "grass"  # Change this to "grass", "forest", or "heterogeneous"

"""
Read base .ini file for case settings.
"""
ini = mht.Read_namelist("plume_chem.ini.base")


"""
Create case input.
"""
if sw_chemistry:

    # Read TUV output table.
    # There is a total of 24 hour available, generated for the
    # Jaenschwalde power plant on 23/05/2022.
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

    # NOTE: `.loc` is value based, not on index.
    tuv = tuv.loc[start_hour:end_hour]

    # Convert to seconds, and subtract starting time.
    tuv.index *= 3600
    tuv.index -= tuv.index.values[0]

    # Emissions (?)
    emi_no = np.zeros(tuv.index.size)
    emi_isop = np.zeros(tuv.index.size)

    xmnh3 = 17.031;
    xmair = 28.9647;
    xmair_i = 1.0 / xmair;
    rho = 1.2658
    c_ug = (1.0e9) * rho * xmnh3 * xmair_i;

    # Concentrations, for now constant with height.
    # species = {"nh3": 0.0} 
    # species = {"nh3": 7.18221-9} # Equivalent to 5 ug/m3 in T=25 C and P=101325 Pa
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
stretch_factor = 1.6  # This gives ~2.5m near surface, ~20m aloft for 384 layers

z, zh = stretched_vertical_grid(z_bottom, z_top, n_layers, stretch_factor)

# Calculate dz for informational purposes
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


# Vertical profiles.
thl = profile(zi=600, v_bulk=291,   dv=2,     gamma_v=0.006)
qt  = profile(zi=600, v_bulk=7.5e-3, dv=-0.3e-3, gamma_v=-0.002e-3, clip_at_zero=True)

u = np.ones(ktot) * 5
nudgefac = np.ones(ktot) / 10800

time = np.linspace(start_hour*3600, end_hour*3600, 97)
hour_of_day = (time / 3600) % 24
# daytime_mask = (hour_of_day >= 6) & (hour_of_day <= 18)
daytime_mask = (hour_of_day >= sunrise_hour) & (hour_of_day <= sunset_hour)

# Heat fluxes should also use hour_of_day for consistency:
# wthl = 0.2 * np.sin(np.pi * hour_of_day / 24)
wthl = 0.2 * np.sin((2.0 * np.pi /24) * (hour_of_day - 6.0))
# wqt = 8e-5 * np.sin(np.pi * hour_of_day / 24)
wqt = 8e-5 * np.maximum(0 , np.sin((2.0 * np.pi /24) * (hour_of_day - 6.0)))

##########################################################
# Surface radiation (only used with land-surface enabled).
##########################################################

max_rad = 650

# Initialize radiation arrays
sw_flux_dn = np.zeros_like(time)
# sw_flux_dn[daytime_mask] = max_rad * np.sin(np.pi * (hour_of_day[daytime_mask] - 6) / 12)
sw_flux_dn[daytime_mask] = max_rad * np.sin(np.pi * (hour_of_day[daytime_mask] - sunrise_hour) / (sunset_hour - sunrise_hour))

#sw_flux_dn = max_rad * np.sin(np.pi * (time-t0) / td)
sw_flux_dn[sw_flux_dn < 0] = 0
sw_flux_up = 0.2 * sw_flux_dn

lw_flux_dn = np.ones_like(sw_flux_dn) * 334
lw_flux_up = np.ones_like(sw_flux_dn) * 472

################################
#Write input NetCDF file.
################################

# Defines helper function add_nc_var to add variables to NetCDF file:
def add_nc_var(name, dims, nc, data):
    if dims is None:
        var = nc.createVariable(name, np.float64)
    else:
        var = nc.createVariable(name, np.float64, dims)
    var[:] = data

# Creates "plume_chem_input.nc" file
nc_file = nc.Dataset("plume_chem_input.nc", mode="w", datamodel="NETCDF4", clobber=True)

# Add start_hour as a variable
add_nc_var("start_hour", None, nc_file, start_hour)  # None means it's a scalar variable without dimensions

add_nc_var("t_sunrise", None, nc_file, sunrise_hour)
add_nc_var("t_sunset", None, nc_file, sunset_hour)

# add_nc_var("max_rad", None, nc_file, max_rad)  # None means scalar variable

###############################
# Sets up dimensions and groups
###############################

# "z" dimension for vertical levels:
nc_file.createDimension("z", ktot)
add_nc_var("z", ("z"), nc_file, z)

# Atmospheric input.
# ("init" group for initial atmospheric conditions)
nc_group_init = nc_file.createGroup("init")

add_nc_var("u", ("z"), nc_group_init, u)
add_nc_var("thl", ("z"), nc_group_init, thl)
add_nc_var("qt", ("z"), nc_group_init, qt)
add_nc_var("nudgefac", ("z"), nc_group_init, nudgefac)
add_nc_var("qt_nudge", ("z"), nc_group_init, qt)
#add_nc_var("co2", ("z"), nc_group_init, co2)
#add_nc_var("co2_inflow", ("z"), nc_group_init, co2)

#("timedep" group for time-dependent surface conditions)
nc_tdep = nc_file.createGroup("timedep");
nc_tdep.createDimension("time_surface", time.size)

add_nc_var("time_surface", ("time_surface"), nc_tdep, time-time[0])
add_nc_var("thl_sbot", ("time_surface"), nc_tdep, wthl)
add_nc_var("qt_sbot", ("time_surface"), nc_tdep, wqt)

if (sw_chemistry):
    # Chemistry input.
    # "timedep_chem" group (if chemistry enabled)
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

    # Add flux_nh3 variable
    add_nc_var("flux_nh3", ("z"), nc_group_init, np.zeros(ktot, dtype=np.float64))
    add_nc_var("flux_inst", ("z"), nc_group_init, np.zeros(ktot, dtype=np.float64))

if (sw_land_surface):
    # "soil" group (if land surface enabled)
    nc_soil = nc_file.createGroup("soil")
    nc_soil.createDimension("z", 4)
    add_nc_var("z", ("z"), nc_soil, np.array([-1.945, -0.64, -0.175, -0.035]))

    add_nc_var("theta_soil", ("z"), nc_soil, np.array([0.34, 0.25, 0.21, 0.18]))
    add_nc_var("t_soil", ("z"), nc_soil, np.array([282, 287, 290, 286]))
    add_nc_var("index_soil", ("z"), nc_soil, np.ones(4) * 2)
    #add_nc_var("root_frac", ("z"), nc_soil, np.array([0.05, 0.3, 0.4, 0.25]))

    # Add idealized (prescribed) radiation.
    add_nc_var("sw_flux_dn", ("time_surface"), nc_tdep, sw_flux_dn)
    add_nc_var("sw_flux_up", ("time_surface"), nc_tdep, sw_flux_up)
    add_nc_var("lw_flux_dn", ("time_surface"), nc_tdep, lw_flux_dn)
    add_nc_var("lw_flux_up", ("time_surface"), nc_tdep, lw_flux_up)

nc_file.close()

"""
Define emissions.
"""
# Coordinates of central cooling tower (m):
# Coordinates of central cooling tower (m):
x0 = 400
y0 = ysize/2.
z0 = 0

# # Std-dev of plume widths:
# sigma_x = 25
# sigma_y = 25
sigma_x = (xsize/itot)*0.5
sigma_y = (ysize/jtot)*0.5
sigma_z = (xsize/itot)*0.5

strength_nh3  = (0.01e-3) / 17.031  # 0.01 g/s (Livestock farm)
# strength_nh3  = 0.0

emi = hlp.Emissions()

y_offsets = [-28, -9, 9, 18]  # Specified y-offsets

for y_offset in y_offsets:
    x = x0
    y = y0 + y_offset
    z = z0

    # Only adding NH3 as the source as requested
    if (sw_chemistry):
        emi.add('nh3', strength_nh3, True, x, y, z, sigma_x, sigma_y, sigma_z)

##############################################################################################
##Create heterogeneous land-surface with three distinct regions (grass_upwind, forest, grass_downwind).
##############################################################################################

if (sw_land_surface):
    # Define block sizes for x and y directions
    blocksize_i = 1  # Size of block in x direction (in grid points)
    blocksize_j = 1  # Size of block in y direction (in grid points)

    # Calculate size of each block in total grid points
    block_points = blocksize_i * blocksize_j  # Total grid points in one block

    # Calculate total number of possible regions in each direction
    region_sizex = itot // blocksize_i  # Number of possible regions in x direction
    region_sizey = jtot // blocksize_j  # Number of possible regions in y direction

    # Print information about block structure
    print(f"\nBlock structure:")
    print(f"Each block is {blocksize_i}×{blocksize_j} grid points")
    print(f"Maximum possible regions: {region_sizex} (x) × {region_sizey} (y)")
    print(f"Selected land surface type: {land_surface_type}")

    # Define boundaries based on land surface type
    if land_surface_type == "heterogeneous":
        # Original heterogeneous setup
        grass_upwind_end = int((1400/dx) // blocksize_i) * blocksize_i
        forest_end = int((1400/dx + 5000/dx) // blocksize_i) * blocksize_i
    elif land_surface_type == "grass":
        # Pure grass: no forest region
        grass_upwind_end = itot  # All grass_upwind
        forest_end = itot        # No forest
    elif land_surface_type == "forest":
        # Pure forest: no grass regions
        grass_upwind_end = 0     # No grass_upwind
        forest_end = itot        # All forest
    else:
        raise ValueError(f"Unknown land_surface_type: {land_surface_type}. Use 'grass', 'forest', or 'heterogeneous'")

    # Print information about domain setup
    print(f"\nDomain setup (aligned to block boundaries):")
    print(f"Grass_upwind region: 0 to {grass_upwind_end*dx}m (0 to {grass_upwind_end} points)")
    print(f"Forest region: {grass_upwind_end*dx}m to {forest_end*dx}m ({grass_upwind_end} to {forest_end} points)")
    print(f"Grass_downwind region: {forest_end*dx}m to {itot*dx}m ({forest_end} to {itot} points)")

    ##########################################################
    # Create and save mask files according to MicroHH format
    ##########################################################

    # Initialize only the masks that will be used
    if land_surface_type == "grass":
        # Only create grass mask covering entire domain
        mask_grass_upwind = np.ones((jtot, itot), dtype=float_type)
        mask_grass_upwind.tofile("grass_upwind.0000000")
        print("Created: grass_upwind.0000000 (entire domain)")

    elif land_surface_type == "forest":
        # Only create forest mask covering entire domain
        mask_forest = np.ones((jtot, itot), dtype=float_type)
        mask_forest.tofile("forest.0000000")
        print("Created: forest.0000000 (entire domain)")

    elif land_surface_type == "heterogeneous":
        # Create all three masks with original boundaries
        mask_grass_upwind = np.zeros((jtot, itot), dtype=float_type)
        mask_forest = np.zeros((jtot, itot), dtype=float_type)
        mask_grass_downwind = np.zeros((jtot, itot), dtype=float_type)

        # Set the regions according to specified boundaries using block-wise assignment
        for j in range(0, jtot, blocksize_j):
            j_slice = slice(j, j + blocksize_j)

            # Assign grass_upwind region blocks
            for i in range(0, grass_upwind_end, blocksize_i):
                i_slice = slice(i, i + blocksize_i)
                mask_grass_upwind[j_slice, i_slice] = 1

            # Assign forest region blocks
            for i in range(grass_upwind_end, forest_end, blocksize_i):
                i_slice = slice(i, i + blocksize_i)
                mask_forest[j_slice, i_slice] = 1

            # Assign grass_downwind region blocks
            for i in range(forest_end, itot, blocksize_i):
                i_slice = slice(i, i + blocksize_i)
                mask_grass_downwind[j_slice, i_slice] = 1

        # Save all three mask files
        mask_grass_upwind.tofile("grass_upwind.0000000")
        mask_forest.tofile("forest.0000000")
        mask_grass_downwind.tofile("grass_downwind.0000000")
        print("Created: grass_upwind.0000000, forest.0000000, grass_downwind.0000000")

    # Create mask list for statistics based on land surface type
    if land_surface_type == "heterogeneous":
        mask_list = "grass_upwind,forest,grass_downwind"
    elif land_surface_type == "grass":
        mask_list = "grass_upwind"  # Only grass mask is active
    elif land_surface_type == "forest":
        mask_list = "forest"  # Only forest mask is active

    ##########################################################
    # Initialize and setup Land Surface Model
    ##########################################################

    # Initialize LSM
    # ls = LSM_input(itot, jtot, 4, sw_water=True, TF=float_type, debug=True, exclude_fields=['z0m', 'z0h'])
    ls = LSM_input(itot, jtot, 4, sw_water=True, TF=float_type, debug=True)

    # Create boolean masks for LSM setup based on land surface type
    if land_surface_type == "grass":
        # Pure grass: no forest anywhere
        mask_forest_bool = np.zeros((jtot, itot), dtype=bool)
        print("LSM setup: Pure grass domain")

    elif land_surface_type == "forest":
        # Pure forest: forest everywhere
        mask_forest_bool = np.ones((jtot, itot), dtype=bool)
        print("LSM setup: Pure forest domain")

    else:  # heterogeneous
        # Original heterogeneous setup
        mask_forest_bool = np.zeros((jtot, itot), dtype=bool)
        mask_forest_bool[:, grass_upwind_end:forest_end] = True
        print("LSM setup: Heterogeneous domain (grass-forest-grass)")

    # Create and set roughness length fields
    z0m = np.zeros((jtot, itot), dtype=float_type)
    z0h = np.zeros((jtot, itot), dtype=float_type)

    # Set roughness lengths for forest
    z0m[mask_forest_bool] = 0.75    # Momentum roughness length for forest
    z0h[mask_forest_bool] = 0.075   # Heat roughness length for forest

    # Set roughness lengths for grass (both in and out regions)
    z0m[~mask_forest_bool] = 0.03   # Momentum roughness length for grass
    z0h[~mask_forest_bool] = 0.003 # Heat roughness length for grass

    # Set z0m and z0h in LSM
    ls["z0m"][:,:] = z0m
    ls["z0h"][:,:] = z0h

    # Root distribution by layer
    root_frac = np.zeros((4, jtot, itot), dtype=float_type)

    # Set root distribution values
    forest_values = [0.24, 0.38, 0.31, 0.07]  # Layers 1-4 for forest
    grass_values = [0.35, 0.38, 0.23, 0.04]   # Layers 1-4 for grass

    # Apply values based on mask
    for layer in range(4):
        for j in range(jtot):
            for i in range(itot):
                if mask_forest_bool[j, i]:  # If forest
                    root_frac[layer, j, i] = forest_values[layer]
                else:  # If grassland
                    root_frac[layer, j, i] = grass_values[layer]

    # Set the root_frac in the land surface model
    ls["root_frac"][:,:,:] = root_frac

    # Save binary files
    z0m.tofile("z0m.0000000")
    z0h.tofile("z0h.0000000")

    def set_value(variable, forest, grass):
        ls[variable][mask_forest_bool] = forest      # Forest values
        ls[variable][~mask_forest_bool] = grass      # Grass values (both in and out)

    # Set surface properties for forest and grass regions
    set_value("c_veg", forest=1.0, grass=1.0)      # Vegetation fraction
    set_value("lai", forest=4.0, grass=3.1)        # Leaf Area Index
    set_value("water_mask", forest=0, grass=0)     # No water surfaces

    ##########################################################
    # Set constant properties for all surfaces
    ##########################################################

     # Surface parameters
    #ls["gD"][:,:] = 0.00              # Vegetation water stress parameter(1/hPa)
    set_value("gD", forest=0.03, grass=0.0)
    #ls["rs_veg_min"][:,:] =100        # Minimum canopy surface resistance
    set_value("rs_veg_min", forest=250, grass=100)
    ls["rs_soil_min"][:,:] = 50      # Minimum soil surface resistance
    ls["lambda_stable"][:,:] = 10     # Skin conductivity for stable conditions (W/m²/K)
    ls["lambda_unstable"][:,:] = 10   # Skin conductivity for unstable conditions (W/m²/K)
    ls["cs_veg"][:,:] = 0.0           # Vegetation heat capacity
    ls["t_bot_water"][:,:] = 295     # Bottom water temperature (K)

    # Soil properties (4 layers)
    ls["t_soil"][:,:,:] = 290        # Soil temperature
    ls["theta_soil"][:,:,:] = 0.3    # Volumetric soil moisture content (m³/m³)
    ls["index_soil"][:,:,:] = 2      # Soil type index
    ls["root_frac"][:,:,:] = 0.25    # Root fraction distribution in each soil layer

    # Check if all values are set
    ls.check()

    # Save LSM setup
    ls.save_binaries(allow_overwrite=True)
    ls.save_netcdf("lsm_input.nc", allow_overwrite=True)

    ###########################
    #Add settings to .ini file.
    ###########################

    # Sets grid parameters:
    ini["grid"]["itot"] = itot
    ini["grid"]["jtot"] = jtot
    ini["grid"]["ktot"] = ktot

    ini["grid"]["xsize"] = xsize
    ini["grid"]["ysize"] = ysize
    ini["grid"]["zsize"] = zsize

    # Add statistics settings for masks
    ini["stats"]["xymasklist"] = mask_list

    # Handles scalar variables:
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

    # Configures sources and chemistry:
    ini["source"]["sourcelist"] = emi.source_list

    ini["chemistry"]["swchemistry"] = sw_chemistry

    # crosslist = ["thl", "qt", "u", "v", "w", "thl_fluxbot", "qt_fluxbot","flux_nh3","flux_inst"]
    # crosslist = ["u", "thl_fluxbot", "qt_fluxbot", "flux_nh3", "flux_inst"]
    crosslist = ["u", "v", "w", "thl_fluxbot", "qt_fluxbot", "flux_inst",
             "ra", "rb", "obuk", "ustar", "ccomp_tot", "cw", "cstom",
             "csoil_eff", "cw_out", "cstom_out", "csoil_out", "rc_tot", "rc_eff",
             "rh_surface", "T_surface", "flux_nh3", "total_flux_mol_ha", "cstar1",
             "cstar2", "c_grid_closest", "c_target", "c_diff_flux"]

    if (sw_chemistry):
        # Add chemicial species and their vertical integrals.
        crosslist += list(species.keys())
        crosslist += [f"{x}_path" for x in species.keys()]
        crosslist += [f"vd{x}" for x in deposition_species]

    # if (sw_land_surface and sw_chemistry):
    #     # Add deposition for each land-surface tile.
    #     for s in deposition_species:
    #         for t in ["soil", "wet", "veg"]:
    #             crosslist.append(f"vd{s}_{t}")

    # crosslist.append("ra")  # Grid-mean aerodynamic resistance
    # crosslist.append("rb")
    # crosslist.append("obuk")
    # crosslist.append("ustar")
    # crosslist.append("ccomp_tot")  # Grid-mean compensation point

    # # Add new resistance components
    # crosslist.append("cw")        # Grid-mean external leaf resistance
    # crosslist.append("cstom")     # Grid-mean stomatal resistance
    # crosslist.append("csoil_eff") # Grid-mean soil effective resistance

    # # Add new compensation points
    # crosslist.append("cw_out")    # Grid-mean external leaf compensation point
    # crosslist.append("cstom_out") # Grid-mean stomatal compensation point
    # crosslist.append("csoil_out") # Grid-mean soil compensation point
    # crosslist.append("rc_tot")
    # crosslist.append("rc_eff")

    # ### Add surface parameters for each land-surface tile
    # for t in ["soil", "wet", "veg"]:
    #     crosslist.append(f"ra_{t}")
    #     crosslist.append(f"rb_{t}")
    #     crosslist.append(f"obuk_{t}")
    #     crosslist.append(f"ustar_{t}")
    #     crosslist.append(f"cw_{t}")
    #     crosslist.append(f"cstom_{t}")
    #     crosslist.append(f"csoil_eff_{t}")
    #     crosslist.append(f"cw_out_{t}")
    #     crosslist.append(f"cstom_out_{t}")
    #     crosslist.append(f"csoil_out_{t}")
    #     crosslist.append(f"rc_tot_{t}")
    #     crosslist.append(f"rc_eff_{t}")

    ## Configures boundary conditions based on land-surface flag:
    # With land-surface:
    if (sw_land_surface):
        ini["boundary"]["swboundary"] = "surface_lsm"
        ini["boundary"]["sbcbot"] = "flux"
        ini["boundary"]["sbot"] = "0"
        ini["boundary"]["thl"] = "dirichlet"
        ini["boundary"]["qt"] = "dirichlet"
        ini["boundary"]["swtimedep"] = False
        ini["boundary"]["timedeplist"] = "empty"

        ini["radiation"]["swradiation"] = "prescribed"

    # Without land-surface (Radiation source is off):
    else:
        ini["boundary"]["swboundary"] = "surface"
        ini["boundary"]["sbcbot"] = "flux"
        ini["boundary"]["swtimedep"] = True
        ini["boundary"]["timedeplist"] = ["thl_sbot", "qt_sbot"]

        ini["radiation"]["swradiation"] = False


    ## Sets deposition settings:
    #  if BOTH chemistry AND land surface are enabled, turns ON deposition in the model!
    if (sw_chemistry and sw_land_surface):
        ini["deposition"]["swdeposition"] = True

    # If either one or both are disabled, turns OFF deposition in the model!
    else:
        ini["deposition"]["swdeposition"] = False

    # Configures cross-section output and source parameters:
    ini["cross"]["crosslist"] = crosslist
    ini["cross"]["xz"] = ysize/2

    # Adds emission source locations and parameters:
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

    #  saves the configuration:
    ini.save("plume_chem.ini", allow_overwrite=True)

    print(f"First cell height: {dz[0]:.2f}m")
    print(f"Last cell height: {dz[-1]:.2f}m")
