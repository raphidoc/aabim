The structure of this software is described as follow:

1. ancillary
For a description of ancillary data see: https://oceancolor.gsfc.nasa.gov/resources/docs/ancillary/

2. converter
Inspired from the excellent ACOLITE software (https://github.com/acolite/acolite) this folder contains the code to convert input remote sensing image format to the standard format use by this software. This allow to create a signle entry point for any number of existing and future remote sensing image format. It make the software easily extensible.

3. image
Contain the code defining the aambi.image class. This class is the fundation on wich other class and method will work. It stores information such as the input file path, xarray dataset, dimensions, coordinates in a way that is meaningfull for optical remote sensing application.

4. model
This folder contains the atmospheric and aquatic models written in C++ and python wrappers.

5. inversion
This folder contains the code use to parametrize the inversion of the atmospheric and aquatic models. The inversion is implemented in STAN, the code in this folder create all the objects that STAN needs to run and parse the results.

6. data
This folder contains the static data used by the software, such as the solar irradiance spectrum and the atmospheric LUTs. The code to create the LUT is also present but depend on the `j6s` python package to run.

7. utils
Utilities and helper functions for a variety of task.

8. experiment
This folder contains various experiment, notably those made with the 6S code.