XRayPanorama
============

This simple program is adapted from 
[07_registration_application](https://github.com/SimpleITK/TUTORIAL/blob/main/07_registration_application.ipynb)
in the [SimpleITK Tutorial](https://github.com/SimpleITK/TUTORIAL). 
Compared to the tutorial version, this adaptation has some additional functionalities:

* The program is easily accessed from command line and is intuitive to use.
* Users have the option to interactively crop the images to remove artifact on the top and bottom edges of the images.
* The program can automatically match the appearances of the three images to account for intensity differences
due to different exposure settings when the images were taken.
* To calculate the HKA angle directly in the program,
users will be prompted to specify three points on the composite image.
The HKA angle will then be calculated and displayed for the user.
* A separate program is provided to directly calculate the HKA angle
when a composite image is already present.


Dependencies
------------
* SimpleITK
* NumPy
* Matplotlib
* Pydicom
* OpenCV

Usage
-----
Given the hip, knee, and ankle images of the same subject
saved as DICOM files `./hip.dcm`, `./knee.dcm`, and `./ankle.dcm`, respectively,
the following script stitches the three images together using an **exploration and exploitation** approach 
and save the composite image as DICOM file `composite.dcm` 
under the directory specified by `--output` or `-o`:

    python main.py -p ./hip.dcm -k ./knee.dcm -a ./ankle.dcm -o ./output

The next section gives an overview of the pipeline of the program. 
Note that the numbers here correspond to the numbered steps in the source code if you ever wish to take a look.

1. The three images are displayed side-by-side for the user 
to visually inspect the images and decide whether to crop the images.
2. After closing the window from step 1, the user is asked whether they wish to crop the images (default is No). 
If Yes, on each image, the user is asked to specify the top and bottom boundaries of the cropped image. 
The specified boundaries are confirmed by pressing Enter.
The cropped images will be displayed for visual inspection 
and saved under the directory specified by `--output` or `-o`.
3. In step 3, three images are stitched using the **exploration and exploitation** approach. 
The resulting composite image is displayed for visual inspection.
4. After closing the window from step 3, 
the user is asked whether they wish to match the appearances of the three images (default is No). 
If Yes, the three images are post-processed using the statistics
calculated from the overlapping regions of the three images. 
The final composite image is displayed for visual inspection and saved as DICOM file `composite.dcm`
under the directory specified by `--output` or `-o`.
5. The user will then be asked whether they wish to calculate the HKA angle.
If Yes, the user will be further asked to specify three points using their left button.
The HKA angle will be displayed after all three points are specified.
To exit the program, press Enter.

When a composite image is already present, 
the user can optionally use the program `HKA_angle_calc.py` instead:

    python HKA_angle_calc.py ./output/composite.dcm

where `./output/composite.dcm` is the path to the composite image.




