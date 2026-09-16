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

1. In VS Code, run **Dev Containers: Rebuild and Reopen in Container**. This creates a containerized Linux environment:

    - Based on `continuumio/miniconda3`.
    - Creates and automatically activates the `mhm-dev` conda environment with all required build dependencies: compilers (gfortran/gcc/g++), CMake, Make, Ninja, pkg-config, NetCDF-Fortran, fypp, and Git.
    - Sets `CMAKE_PREFIX_PATH` to the conda environment so CMake resolves NetCDF-Fortran automatically.
    - Configures CMake Tools to use Ninja, write to `./build`, and enable OpenMP.
    - Installs the Fortran linter, CMake Tools, and Python VS Code extensions.

    No package installation or manual conda activation is required after the container is built. The mHM executable is not built as part of container creation, so compile it once from a terminal in the container:

    ```bash
    cmake -S . -B build -G Ninja -DCMAKE_WITH_OpenMP=ON
    cmake --build build
    ```

    `-B build` creates an out-of-source build, `-G Ninja` selects Ninja, and `-DCMAKE_WITH_OpenMP=ON` requires OpenMP support. Configuration fails instead of silently producing a serial build if OpenMP is unavailable.

2. Confirm OpenMP is enabled in the configured build:

    ```bash
    grep '^CMAKE_WITH_OpenMP:BOOL=ON$' build/CMakeCache.txt
    ```

3. Run mHM on the test domain with the desired number of CPU threads (for example, four):

    ```bash
    cd /workspace/test_domain
    OMP_NUM_THREADS=4 ../build/mhm
    ```

    At startup, mHM reports `OpenMP used.` and the number of threads. Adjust `OMP_NUM_THREADS` for the CPU resources available to the container.

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
