# MicroHH-DEPAC v1.0.0: Input data, configuration files, and analysis scripts

This repository contains the input data, configuration files, selected model
output, and analysis scripts accompanying the manuscript:

Rashidi, M., van Zanten, M. C., and Krol, M. C.: MicroHH-DEPAC v1.0:
large-eddy simulation of bidirectional ammonia surface exchange with a
reference-height formulation, Geoscientific Model Development, 2026
(in review).

---

## Model code

MicroHH-DEPAC is a fork of the MicroHH large-eddy simulation model
(van Heerwaarden et al., 2017) with the DEPAC dry deposition module
(van Zanten et al., 2010) integrated. The model source code, build
instructions, and installation are maintained in a separate repository:

    https://github.com/mohammadr89/microdepac

The exact version of the code used to produce the results in this paper is
archived at:

    https://doi.org/10.5281/zenodo.22131443  (release v1.0.0-GMD)

This repository (input data, configuration files, and analysis scripts) is
archived separately at:

    https://doi.org/10.5281/zenodo.22230061

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)

Original MicroHH copyright: Chiel van Heerwaarden, Thijs Heus, and MicroHH
contributors (base code: https://github.com/microhh/microhh). The DEPAC
module was adapted from the implementation by Geers et al. (2025), obtained
from https://github.com/dalesteam/dales/blob/ruisdael_deposition/src/le_drydepos_gas_depac.f90
(commit c8d343e). DEPAC coupling additions: Mohamadreza Rashidi (2025).

---

## Repository structure

    cases/
        grassland/
            bg_bidir_dz04/                    Case 1, Scenario A: background NH3 only
            bg_onedir_dz04/                   Case 1, Scenario A: background only, unidirectional
            ps_onedir_dz04/                   Case 1, Scenario B: point source, unidirectional
            ps_bidir_dz04/                    Case 1, Scenario B: point source, bidirectional
            psbg_bidir_dz04/                  Case 1, Scenario C: point source + background
        forest/
            bg_bidir_dz04/                     Case 2, Scenario A: dz = 4 m
            bg_bidir_dz08/                     Case 2, Scenario A: dz = 8 m
            bg_bidir_dz12/                     Case 2, Scenario A: dz = 12 m
            bg_bidir_dz16/                     Case 2, Scenario A: dz = 16 m
            bg_bidir_no_target_height_dz04/    Case 2, Scenario A: dz = 4 m, lowest-level sampling
            bg_onedir_dz04/                    Case 2, Scenario A: unidirectional, reference-height sampling
            bg_onedir_no_target_height_dz04/   Case 2, Scenario A: unidirectional, lowest-level sampling
    figures/                   Final figures as they appear in the manuscript
    scripts/                   Python scripts to reproduce Figs. 3-7, D1, E1, and the
                                deposition-flux comparisons in Sect. 5.2 and 6.1
    shared_inputs/             Common files copied into a case before running
    LICENSE                    Creative Commons Attribution 4.0 International
    README.md                  Documentation and instructions

---

## Shared input files

To avoid duplicate copies of files that are identical for all simulation
cases, common input and utility files are stored once in the
`shared_inputs/` directory:

    shared_inputs/
        cloud_coefficients_lw.nc
        cloud_coefficients_sw.nc
        coefficients_lw.nc
        coefficients_sw.nc
        constants.py
        helpers.py
        lsm_input.py
        microhh_tools.py
        plume_chem_tuv_input.txt
        plume_chem_tuv_output.txt
        van_genuchten_parameters.nc

The radiation coefficient files are part of the RRTMGP radiation scheme:

    https://github.com/earth-system-radiation/rte-rrtmgp

The file `van_genuchten_parameters.nc` contains the soil hydraulic parameters
used by the MicroHH land surface model.

The file `lsm_input.py` is a general MicroHH utility module used by the
case-specific `plume_chem_input.py` scripts to generate land surface model
input fields.

The files `constants.py` and `helpers.py` provide physical constants and
helper functions (saturation vapor pressure, emission-source configuration)
used by the case-specific `plume_chem_input.py` scripts.

The file `microhh_tools.py` is a general MicroHH Python utility module
(namelist I/O, statistics and binary-field readers) used by the
`plume_chem_input.py` scripts and, optionally, for post-processing.

The files `plume_chem_tuv_input.txt` and `plume_chem_tuv_output.txt` contain
the TUV photolysis input and output data used by the chemistry configuration.

Before running a simulation, copy the files from `shared_inputs/` into the
selected case directory. For example, from the root directory of this
repository:

    cp shared_inputs/* cases/grassland/bg_bidir_dz04/

---

## Contents of each case directory

    plume_chem.ini.base         Base configuration template for MicroHH
    plume_chem_input.py         Generates the final .ini file and input data

Selected NetCDF output files used by the analysis scripts (e.g.
`flux_inst.xy.nc`, `nh3_first_level_xy.nc`, `ccomp_tot.xy.nc`) are included
in the case directories that require them; see the figure/script mapping
below for which files each script reads.

---

## How to run a simulation

1. Clone and compile MicroHH-DEPAC from:

       https://github.com/mohammadr89/microdepac
       (archived version: https://doi.org/10.5281/zenodo.22131443)

   following the compilation instructions at:

       https://microhh.readthedocs.io/en/latest/getting_started/code_and_compilation.html

2. Copy the common input files into the desired case directory. For example:

       cp shared_inputs/* cases/grassland/bg_bidir_dz04/

3. Navigate to the desired case directory:

       cd cases/grassland/bg_bidir_dz04/

4. Generate the input files:

       python plume_chem_input.py

5. Run MicroHH:

       mpirun -n N microhh init plume_chem
       mpirun -n N microhh run plume_chem

   where N is the number of MPI tasks. The simulations in this paper were run
   on the Snellius supercomputer (NWO, the Netherlands). The finest-resolution
   case (dz = 4 m, 24-hour simulation) required approximately 5 x 10^4 CPU
   hours.

---

## How to reproduce the figures and reported numbers

Final PNG/PDF versions of all manuscript figures are included in the
`figures/` directory. Figures 1 and 2 are static schematics reproduced or
adapted from other publications (van Zanten et al., 2010; Basu and Lacser,
2017) and have no generating script.

Run each script from the root directory of this repository.

    Script                                    Produces                    Cases used
    -------------------------------------------------------------------------------------------------------
    f03.py                                    Fig. 3                      grassland/ps_onedir_dz04
                                                                          grassland/psbg_bidir_dz04
    f04_a.py                                  Fig. 4a                     grassland/bg_bidir_dz04
    f04_b.py                                  Fig. 4b                     grassland/bg_bidir_dz04
    f05.py                                    Fig. 5 (a, b)               grassland/psbg_bidir_dz04
    f06.py                                    Fig. 6                      grassland/ps_bidir_dz04
    f07.py                                    Fig. 7                      forest/bg_bidir_dz04
                                                                          forest/bg_bidir_dz08
                                                                          forest/bg_bidir_dz12
                                                                          forest/bg_bidir_dz16
    fD1.py                                    Fig. D1 (Appendix D)        theoretical illustration, no case output
    fE1.py                                    Fig. E1 (Appendix E)        prescribed radiative forcing, no case output
    analysis_forest_refheight_deposition.py   Sect. 5.2					  forest/bg_onedir_dz04
                                                                          forest/bg_onedir_no_target_height_dz04
    analysis_forest_bidir_deposition.py       Sect. 5.2                   forest/bg_bidir_dz04
    analysis_forest_grassland_deposition.py   Sect. 6.1                   grassland/bg_onedir_dz04
                                                                          forest/bg_onedir_dz04
                                                                          forest/bg_onedir_no_target_height_dz04																		  

### Python dependencies

    Python >= 3.8
    numpy
    matplotlib
    netCDF4
    scipy
    cmcrameri

Install with:

    pip install numpy matplotlib netCDF4 scipy cmcrameri

---

## References

van Heerwaarden, C. C., van Stratum, B. J. H., Heus, T., Gibbs, J. A.,
Fedorovich, E., and Mellado, J. P.: MicroHH 1.0: a computational fluid
dynamics code for direct numerical simulation and large-eddy simulation
of atmospheric boundary layer flows, Geosci. Model Dev., 10, 3145-3165,
https://doi.org/10.5194/gmd-10-3145-2017, 2017.

van Zanten, M. C., Sauter, F. J., Kruit, R. J. W., and van Jaarsveld, J. A.:
Description of the DEPAC Module, 2010.

Geers, L., Janssen, R., Thorkelsdottir, G., Vila-Guerau De Arellano, J.,
and Schaap, M.: Implementation of a Dry Deposition Module (DEPAC v3.11)
in a Large Eddy Simulation Code (DALES v4.4),
https://doi.org/10.5194/egusphere-2025-426, 2025.

Basu, S. and Lacser, A.: A Cautionary Note on the Use of Monin-Obukhov
Similarity Theory in Very High-Resolution Large-Eddy Simulations,
Boundary-Layer Meteorology, 163, 351-355,
https://doi.org/10.1007/s10546-016-0225-y, 2017.

---

## Contact

Mohamadreza Rashidi
Meteorology and Air Quality Group
Wageningen University & Research
mohamadreza.rashidi@wur.nl

