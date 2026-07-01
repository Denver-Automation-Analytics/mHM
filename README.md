# The mesoscale Hydrological Model -- mHM

- The latest mHM release can be found [here][0].
- The changelog can be found in the file [RELEASES][3].
- General information can be found on the [mHM website](https://mhm-ufz.org/).
- The mHM comes with a [LICENSE][6] agreement, this includes also the GNU Lesser General Public License.
- There is a list of [publications using mHM][7].

**Please note:** The [GitLab repository](https://git.ufz.de/mhm/mhm) grants read access to the code.
If you like to contribute to the code, please contact [mhm-admin@ufz.de](mailto:mhm-admin@ufz.de).

## Documentation

The online documentation for mHM can be found here (pdf versions are provided there as well):
- stable: https://mhm.pages.ufz.de/mhm
- latest: https://mhm.pages.ufz.de/mhm/latest

## Cite as

Please refer to the main model by citing Samaniego et al. (2010) and Kumar et al. (2013):

> Samaniego L., R. Kumar, S. Attinger (2010): Multiscale parameter regionalization of a grid-based hydrologic model at the mesoscale. Water Resour. Res., 46,W05523, doi:10.1029/2008WR007327, http://onlinelibrary.wiley.com/doi/10.1029/2008WR007327/abstract

> Kumar, R., L. Samaniego, and S. Attinger (2013): Implications of distributed hydrologic model parameterization on water fluxes at multiple scales and locations, Water Resour. Res., 49, doi:10.1029/2012WR012195, http://onlinelibrary.wiley.com/doi/10.1029/2012WR012195/abstract

The model code can be generally cited as:

> **mHM:** Luis Samaniego et al., mesoscale Hydrologic Model. Zenodo. doi:10.5281/zenodo.1069202, https://doi.org/10.5281/zenodo.1069202

To cite a certain version, have a look at the [Zenodo site][10].

## Install

mHM can be compiled with cmake. See more details under [cmake manual][9].
See also the [documentation][5] for detailed instructions to setup mHM.


## Quick start

1. Compile mHM
2. Run mHM on the test domains with the command `./mhm`, which uses settings from [mhm.nml](mhm.nml).
3. Explore the results in the [output directory](test_domain/), e.g. by using the NetCDF viewer `ncview`.


## Devcontainer

1. Build the devcontainer in VSCode. This creates a containerized linux environment: 

    - Based on continuumio/miniconda3
    - mhm-dev conda env from conda-forge with all required build deps: compilers (gfortran/gcc/g++), cmake, make ninja, pkg-config, netcdf-fortran, fypp, git.
    - CMAKE_PREFIX_PATH set to the conda env so CMake resolves NetCDF-Fortran automatically
    - Default CMake generator: Ninja, output dir: ./build
    - Pre-installed extensions: Fortran linter, CMake Tools, Python

2. Run the following command in the terminal:

    cmake -S . -B build -G Ninja
    cmake --build build

    - B build tells CMake to put all generated build files in the build directory (out-of-source build).
    - G Ninja selects Ninja as the build system generator.
    - cmake --build build uses the generated build system in build (Ninja here) to actually compile and link targets.

3. Run mHM on the test domain from the terminal:

    cd /workspace/test_domain
    ../build/mhm

4. Run mHM on a pre-calibrated basin from Rokovec et al (2019)

    - Populate the run_mhm_config.json file with the required data paths
    - Run the script run_mhm.py

    What the script does:

    Validates the config — checks all paths exist, dates are valid, gauge files are present, binary is executable
    Creates a temp working dir (or work_dir from config if specified) — the calib_001/ folder is never modified
    Copies mhm_parameter.nml (calibrated parameters) verbatim from calib_001/exe/
    Patches mhm.nml using regex substitution across 22 keys (directories, dates, timestep, time_step_model_inputs, warming_days) and regenerates the &evaluation_gauges block from the gauges list — supports any number of gauges
    Patches mhm_outputs.nml to ensure outputFlxState(11)=.TRUE. (L1_total_runoff)
    Runs the mHM binary via subprocess, streams output, checks for "mHM: Finished!"
    Reports the output path and lists the NetCDF files produced; cleans up the temp dir unless keep_work_dir: true

## License

LGPLv3 (c) 2005-2025 mHM-Developers

[0]: https://git.ufz.de/mhm/mhm/-/releases
[3]: doc/RELEASES.md
[4]: https://git.ufz.de/mhm/mhm/tags/
[5]: https://mhm.pages.ufz.de/mhm
[6]: LICENSE
[7]: https://mhm-ufz.org/about/publications/
[9]: doc/INSTALL.md
[10]: https://zenodo.org/record/3239055
