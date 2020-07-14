**NeutPy**

**Installation**

- **Triangle Installation**

The Triangle 2D mesh generator is required for neutpy mesh generation. This will install Triangle
locally to the neutpy directory. If you imagine using triangle otherwise, consider
following the steps below but with consideration for global installation (e.g., cloning to /opt/ instead
of the neutpy directory). Ensure that you have a C compiler installed.
Download the Triangle zip file from https://www.cs.cmu.edu/~quake/triangle.html or 

`$ cd /your/future/neutpy/home/`

`$ git clone https://github.com/libigl/triangle.git`

`$ cd triangle`

Make your bin directory

`$ mkdir bin`

Read the README file for instructions on how to compile. It's pretty basic. We recommend simply
compiling triangle alone with (using GCC) since we do not use showme.

`$ gcc -O -o bin/triangle triangle.c -lm`

If you want to fully compile, edit the makefile,
noting any special options from the README.

Keep `SRC = ./` and set `BIN = ./bin/`

Make triangle

`$ make`

After triangle is compiled, set executable

`$ cd bin`

`$ sudo chmod +x triangle`

Set link (this allows triangle to be called on command line as triangle)
 
`$ sudo ln -s triangle /usr/local/bin`

**Install NeutPy**

`$ cd /your/future/neutpy/home/`

- **Master branch**

Clone  master branch from github

`$ git clone https://github.com/gt-frc/neutpy.git`

- **Other branches**

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

**Usage**

NeutPy requires 6 input files:

`inputs/toneutprep` is the main input file and sets neutpy in motion with `neutpy_prep('toneutprep')`

Data files:

The data files included follow the GT3 gt3_shotid_timeid_profile.dat convention but can be defined 
differently in your toneutprep file (which can also be named differently)

Ion/Electron density and temperature data are X/Y (normalized rho/value) two-column data. Temperatures are
in keV. Densities should be given in #/m^3. Psi data are non-normalized 3-column R/Z/value data, with R/Z in 
meters.

`inputs/shot_timeid/gt3_shotid_timeid_ne.dat` (Electron density profile)

`inputs/shot_timeid/gt3_shotid_timeid_ni.dat` (Ion density profile)

`inputs/shot_timeid/gt3_shotid_timeid_Te.dat` (Electron temperature profile)

`inputs/shot_timeid/gt3_shotid_timeid_Ti.dat` (Ion temperature profile)

`inputs/shot_timeid/gt3_shotid_timeid_psirz.dat` (Non-normalized psi grid)

`inputs/shot_timeid/gt3_diid_wall.dat` (DIII-D wall coordinates (R/Z))

Output:

NeutPy will generate output in the `outputs` folder. The data are 2-D R/Z

`n_n_slow` - Slow (cold) neutral density 

`n_n_thermal` - Thermal neutral density

`n_n_total` - Total neutral density

`izn_rate_slow` - Ionization rate of cold neutrals

`izn_rate_thermal` - Ionization rate of thermal neutrals

`izn_rate_total` - Total ionization rate

`T_coef.txt` is an intermediate transmission coefficient datafile