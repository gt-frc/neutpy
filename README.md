##NeutPy - A code for the calculation of neutral particle transport in plasmas based on the transmission and escape probability method

NeutPy is the Python 2.x port of GTNEUT, which was written by John Mandrekas.

GTNEUT is a two-dimensional code for the calculation of the transport of neutral particles in fusion plasmas.
It is based on the Transmission and Escape Probabilities (TEP) method and can be considered a computationally efficient
alternative to traditional Monte Carlo methods. The code has been benchmarked extensively against Monte Carlo and
has been used to model the distribution of neutrals in fusion experiments.

The original physics background can be found at

Mandrekas, John. (2004). GTNEUT: A code for the calculation of neutral particle transport in plasmas based on the
    Transmission and Escape Probability method. Computer Physics Communications.
    161. 36-64. 10.1016/j.cpc.2004.04.009.

The original FORTRAN 95 GTNEUT code can be found at The Fusion Research Center GitHub at
https://github.com/gt-frc/GTNEUT

#Installation

- ###Triangle Installation

The Triangle 2D mesh generator is required for NeutPy mesh generation. This guide can be used to install Triangle
locally. If you imagine using triangle otherwise, consider
following the steps below but with consideration for global installation (e.g., cloning to /opt/ instead
of your home directory). Ensure that you have a C compiler installed.
Download the Triangle zip file from https://www.cs.cmu.edu/~quake/triangle.html or 

`$ cd ~`

`$ git clone https://github.com/libigl/triangle.git`

`$ cd triangle`

Make your bin directory

`$ mkdir bin`

Read the README file for instructions on how to compile. It's pretty basic. We recommend simply
compiling triangle alone with (using GCC) since we do not use showme.

`$ gcc -O -o bin/triangle triangle.c -lm`

If you want to fully compile triangle and showme, edit the makefile,
noting any special options from the README.

Keep `SRC = ./` and set `BIN = ./bin/`

Make triangle

`$ make`

After triangle is compiled, set executable

`$ cd bin`

`$ sudo chmod +x triangle`

Set link (this allows triangle to be called on command line as triangle) to /usr/local/bin or some 
other directory on your PATH.
 
`$ sudo ln -s /full/path/to/triangle /usr/local/bin/triangle`

#Install NeutPy

###Using pip

`$ pip install neutpy`

###From GitHub

If you'd like to work on the actual neutpy code, you can `clone` from GitHub:

`$ cd /your/future/neutpy/home/`

- #####Master branch

Clone  master branch from github

`$ git clone https://github.com/gt-frc/neutpy.git`

- #####Other branches

You can clone another branch from github as follows:

`$ git clone -b <branch> https://github.com/gt-frc/neutpy.git`

Enter NeutPy

`$ cd neutpy`

Setup your virtual environment (install virtualenv using apt, yum, etc.)

`$ virtualenv --python=/usr/bin/python2.7 venv`

Activate it

`$ source venv/bin/activate`

Install packages

`$ pip install -r requirements.txt`

###Usage

NeutPy requires 6 input files and 2 configuration files.

Configuration files:

`toneutpy.conf` is the main input file for a shot. This file will change with each shot.

`neutpy.conf` is the main NeutPy configuration file and includes parameters that probably wouldn't change between
shots and other variables. This file needs to be in the current working directory and cannot be renamed. Grab an
 example copy from the FRC GitHub (https://github.com/gt-frc/neutpy/)

Data files:

The names of the data files included follow the GT3 gt3_shotid_timeid_profile.dat convention but can be defined 
differently in your `toneutpy.conf` file.

Ion/Electron density and temperature data are X/Y (normalized rho/value) two-column data. Temperatures are
in keV. Densities should be given in #/m^3. Psi data are non-normalized 3-column R/Z/value data, with R/Z in 
meters.

###Example File Structure

`gt3_shotid_timeid_ne.dat` (Electron density profile)

`gt3_shotid_timeid_ni.dat` (Ion density profile)

`gt3_shotid_timeid_Te.dat` (Electron temperature profile)

`gt3_shotid_timeid_Ti.dat` (Ion temperature profile)

`gt3_shotid_timeid_psirz.dat` (Non-normalized psi grid)

`gt3_diid_wall.dat` (Machine wall coordinates (R/Z))

###Example Usage

Import the neutrals class.
```
>>> from neutpy import neutrals
```
There are three main entry points into NeutPy: `from_file`, `from_mesh`, and `from_gt3`.

####from_file()

```python
>>> neuts = neutrals()
INITIALIZING NEUTPY
>>> neuts.from_file('/relative/path/to/your/toneutpy.conf')
```

where the filename is relative to the CWD.

####from_gt3()

```python
>>> neuts = neutrals()
INITIALIZING NEUTPY
>>> neuts.from_gt3(<GT3 object>)
```
The `from_mesh` method is in development.