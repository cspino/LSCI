App for a laser speckle contrast imaging (LSCI) system.
Compatible with Thorlabs Scientific and Compact USB Cameras, and with Windows.

## Setup
1. Download or clone the repository to your machine
2. Install [ThorCam](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam)
3. Open a terminal window and set the working directory to the repo
```
cd <path to folder where repo was cloned>
```
4. Create a new conda environment from the environment.yml file
```
conda env create -f environment.yml -n <env_name>
```

5. Activate the environment
```
conda activate <env_name>
```

6. Install the thorlabs toolkit (Full procedure on the [Thorlabs GitHub](https://github.com/Thorlabs/Camera_Examples/tree/main/Python))

* If you have not done so already unzip the following folder to an accesible location on your drive. This contains the Camera SDK.
   
   * 32-Bit - C:\Program Files (x86)\Thorlabs\Scientific Imaging\Scientific Camera Support\Scientific Camera Interfaces.zip
   * 64-Bit - C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support\Scientific Camera Interfaces.zip

* To install the Python SDK in your environment, use a package manager such as pip to install from the package file. The zip folder will be found within the location in the last step: \Scientific Camera Interfaces\SDK\Python Toolkit

```
python.exe -m pip install thorlabs_tsi_camera_python_sdk_package.zip
```

* The DLLs are already in the repo, no need to copy them

## Launch the app
Run this command to launch the app
```
python LSCI_app.py
```
