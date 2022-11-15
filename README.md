# rmap
The Rotation Measure Analysis Program - A program designed to calculate Rotation Measures given a set of data from the Canadian Galactic Plane Survey

- Written by Dr. Jo-Anne Brown
- Modified by Cameron Van Eck
- Ported to Python by Dylan Tuttle

## Overview

This program assists the user in analyzing polarisation data from the Canadian
Galactic Plane Survey. There are two main components to this program:

#### 1. Generation of polarised candidate sourcelists (`generate_candidate_sourcelist.py`)
This component compares input mosaic data with the Taylor et. al. 2017 source catalog, and outputs a list of 'candidate sources' (i.e. sources with significant S:N in Stokes I which also have significant S:N in Stokes Q and U) for the purposes of further analysis.

#### 2. Analysis of rotation measures (`calculate_rm.py`)
This component takes the polarised candidate sourcelist and generates a series of plots to assist the user in analyzing the rotation measure data of the polarised sources.

## Installation Instructions

1. [Install Python 3](https://www.python.org/downloads/)
    - IMPORTANT: If you are on Windows, in order to be able to run this program from the command line you must enable the "Add to PATH" checkbox when installing.
    - NOTE: This program was written in Python 3.7, but any version of Python 3 should be able to run it with no issues.

2. Install dependencies using pip
    - This program requires 4 Python packages, which must be installed before the code can be run. pip is an installer program included by default with any installation of Python 3.4 and above. To install any package from [pypi.org](https://pypi.org/) (which as of the writing of this README hosts 415,789 projects), simply open your Terminal/Command Prompt and enter the command `pip install <package>`.
    - If you encounter a 'command not found' error while trying to install these packages, it may be caused by your installation of Python 3 coming with pip3 instead of pip. In this case, simply replace `pip` with `pip3` in any installation commands you invoke.
    - To install the 4 packages this program depends on, you will need to enter the following 4 commands one by one, and follow any subsequent instructions:
    ```
    pip install numpy
	pip install astropy
	pip install matplotlib
	pip install scipy
    ```
    - NOTE: As of the writing of this README, the `astropy` package is unable to be installed on Mac OS through pip. If this fact changes, this README will be updated.

3. Set up data
    - When you clone this project onto your computer, it will have the following directory structure and contain the following files:
    ```
    rm_program/
    |---program_code/
    |   |---main_code/
    |   |   |---subroutines/
    |   |   |   |---array_calculations.py
    |   |   |   |---fits_handling.py
    |   |   |   |---misc_functions.py
    |   |   |---calculate_rm.py
    |   |   |---generate_candidate_sourcelist.py
    |   |---config.py
    |   |---rmap.py
    |---Data/
    |   |---output_data/
    |   |---raw_data/
    |   |---Chi_Table_05.dat
    |   |---Taylor17_CGPS1420catalogue.dat
    |---README (you are here)
    ```
    - The program is set up to collect FITS images from the `raw_data` directory, organized by mosaic. Each mosaic folder must be named after the mosaic, in lowercase with the M removed. For example, the folder containing raw FITS images from MA1 must be named `a1`.
	- The `output_data` folder is where the program will write any files it generates. If the program is run successfully and completely for mosaic, say, MA1, a new folder named `a1` will be created inside `output_data`, and will contain either 2, 3, or 4 `.dat` files:
        - `MA1_Taylor17_candidates.dat` - A table containing the polarised candidate source list
        - `MA1_RMlist_recalc.dat` - A table containing the every source from the candidate source list, along with all of the parameters calculated while running `calculate_rm.py`
        - `MA1_RMlist_recalc_ONLY_GOOD.dat` - A shorter version of the previous table which will only be created if there is at least one source with a source flag of zero and an mChi2 value less than or equal to 5.
        - `MA1_sourcelist_revisit.dat` - A table which will only be created if the user flags any sources with the '1' or 'revisit' flag, for the purposes of running the program again with the pixel flagging feature (NOTE: the pixel flagging feature is not currently included in this version of the program)
    - If you want your data to be stored in another location, or if you already have it stored somewhere and don't want to move it, there is a file in the `program_code` directory called `config.py` which contains the default paths for input data, output data, the Chi table, and the Taylor17 catalogue. Any of these can be changed to the ABSOLUTE(!!) path location where you keep your data. The new locations must still follow the same naming and storage conventions as the default locations.

## Running Instructions

NOTE: This program can not be run from an IDE like PyCharm. It must be run from the command line.

1. In a Terminal/Command Prompt window, navigate to the location of the program_code directory
    - NOTE: This is a necessary step unless you've replaced all of the default paths in `config.py` with your own absolute paths

2. Type the command:
    ```
    python rmap.py
    ```
    - NOTE: I've had issues on Mac OS with this command running the program with the default installation of Python, which is usually some flavour of Python 2. In this case, you have to tell the computer to run the program with your new installation of Python 3. For example, `python3.9 rmap.py`

## Possible future improvements/fixes

- The pixel flagging feature needs to be implemented.
- Polarisation angles in the linear fit plot are not restricted to [0, 180], but the slopes are still correct
- Polarisation angle error bars are too big, but `probfit` is still correct
- Beam FWHM shape acts unpredictably sometimes
- It might be worth it to get it running in an IDE...