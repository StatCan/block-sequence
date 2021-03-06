# Setting Up Environments and NetworkX 

## Python PATH

1.	Ensure that python 3.x is on your device. 
2.	If it is not: 
  a.	Go to python.org 
  b.	Download the latest version (3.7.3) 
  c.	Check box that says “add to PATH” 
3.	To check to make sure it is installed properly, go to your windows powershell then type “py” and it should tell you the 
  recent version you are working with. 
 
## Pip Install 

The python module pip is automatically installed with the new python version. It should run fine as long as when you first 
downloaded python you added it to your PATH.

1.	If you don’t have “numpy” install it now 
    `pip install numpy`
2.	If you don’t have “pandas” install it now 
    `pip install pandas`
3.	If you don’t have “matplotlib” install it now
    `pip install matplotlib`
4.	If you don’t have “Path” install it now 
    `pip install Path`
5.	If you don’t have “itertools” install it now 
    `pip install itertools`
    
## GDAL Installation 

1.	Go to https://www.lfd.uci.edu/~gohlke/pythonlibs/ 
  a.	Download gdal-2.4.1cp31m-win32.whl 
  b.	Once it is finished downloading, copy to C:\ drive 
  c.	In powershell
      `pip install C:\gdal-2.4.1cp31m-win32.whl`

NOTE: If there is an error that says that the whl isn’t supported, you downloaded the wrong .whl file for your device.

2.	Go to where the GDAL WHL saved the folder. In order for GDAL to work properly, the folder paths need to be added to 
the computer's environment manually. 
  a.	Right-click "This PC" and select "Properties"
  b.	Go to “Advanced system settings”
  c.	Go to “environment variables“
  d.	Under “System variables” click PATH and then Edit. Next click “New” and add the folder path to the GDAL folder 
      i.e. C:\GDAL-2.4.1.dist-info then click ok.
  e.	Under “System variables” click NEW. Input:
        Variable Name: GDAL_DATA 
        Variable value: C:\GDAL-2.4.1.dist-info\data
  i.	Under “System variables” click NEW. Input: 
        Variable Name: GDAL_VARIABLE_PATH 
 		    Variable value: C:\GDAL-2.4.1.dist-info\gdalplugins 
  f.	Under “System variables” click NEW. Input:
        Variable Name: GDAL_VERSION 
        Variable value: 2.4.1 
  g.	Click Ok 

NOTE: For the “Variable value” make sure it is your proper path to those folders. Input the “Variable name” as indicated. 

3.	Where you downloaded GDAL.whl file also download Fiona-1.8.6-cp37-cp37m-32.whl. Repeat steps 1 a – c. 
4.	Repeat steps (1 a – c) for geopandas. 
5.	Repeat steps (1 a – c) for pyproj.
6.	Repeat steps (1 a – c) for shapely. 
7.	Repeat steps (1 a – c) for Click. 
8.	After everything is loaded, installed and set up: `pip install networkx`

NOTE: For references on all the requirements needed, please refer to the requirements.txt. 

## Adding Jupyter Python 3 Kernel 

If you are using Jupyter notebook and there is no Python 3 kernel on it, you need to add it now. 

NOTE: If you had to install python 3 make sure that the path is in the environment where we put the GDAL variable information. 
  1.	`py –m pip install ipykernel`
  2.	`py –m pip ipykernel install`

Jupyter kernel Python 3 should appear. 
