Classes
-------

Three classes encapsulate the code logic:

.. toctree::
   :hidden:
   
   imager
   segmentation
   reconstruction
   
- :doc:`imager`: describes an imager and all the parameters related to it. Contains the methods dealing with lens distortions and mis-alignment correction.
- :doc:`segmentation`: contains all the methods needed for the segmentation a sky/cloud image.
- :doc:`reconstruction`: contains all the methods for creating a 3D reconstruction from the base height from two sky pictures taken at the same time.

Sample usage
------------

The file main.py shows a sample usage for the code to generate the reconstruction of cloud base height. Here is a walkthrough the file:

First import the dependencies:

.. code-block:: python

   from imager import Imager
   from reconstruction import Reconstruction
   import numpy as np
   import cv2
   import datetime

Then create the first imager object. For the calibration parameters, please refer to the following publication: `Geo-referencing and stereo calibration of ground-based whole sky imagers using the sun trajectory <http://vintage.winklerbros.net/Publications/igarss2016a.pdf>`_, F. M. Savoy, S. Dev, Y. H. Lee, S. Winkler, Proc. IEEE International Geoscience and Remote Sensing Symposium (IGARSS), Beijing, China, July 10-15, 2016.

.. code-block:: python

   name3 = 'wahrsis3'
   center3 = [1724, 2592]
   radius3 = 1470
   relativePosition3 = np.array([0,0,0])
   calibRot3 = np.array([[0.99555536, 0.09404159, 0.00506982], [-0.09393761, 0.99541774, -0.01786745], [-0.00672686, 0.01731178, 0.99982751]])
   calibTrans3 = np.array([[ 0.00552915], [0.00141732], [0.00553584]])
   longitude3 = '103:40:49.9'
   lattitude3 = '1:20:35'
   altitude3 = 59
   wahrsis3 = Imager(name3, center3, radius3, relativePosition3, calibRot3, calibTrans3, longitude3, lattitude3, altitude3)

The second imager does not need the longitude, lattitude and altitude information.

.. code-block:: python

   name4 = 'wahrsis4'
   center4 = [2000, 2975]
   radius4 = 1665
   relativePosition4 = np.array([-2.334, 101.3731, -8.04])
   calibRot4 = np.array([[0.9710936, -0.23401871, 0.04703662], [0.234924, 0.97190314, -0.01466276], [-0.04228367, 0.02528894, 0.99878553]])
   calibTrans4 = np.array([[-0.00274625], [-0.00316865], [0.00516088]])
   wahrsis4 = Imager(name4, center4, radius4, relativePosition4, calibRot4, calibTrans4)

Then load the LDR images in two arrays:

.. code-block:: python

   images3 = [cv2.imread('wahrsis3/2015-10-29-12-58-01-wahrsis3-low.jpg'),
   	cv2.imread('wahrsis3/2015-10-29-12-58-01-wahrsis3-med.jpg'),
   	cv2.imread('wahrsis3/2015-10-29-12-58-01-wahrsis3-high.jpg')]
   images4 = [cv2.imread('wahrsis4/2015-10-29-12-58-01-wahrsis4-low.jpg'),
   	cv2.imread('wahrsis4/2015-10-29-12-58-01-wahrsis4-med.jpg'),
   	cv2.imread('wahrsis4/2015-10-29-12-58-01-wahrsis4-high.jpg')]

Create the Reconstruction object, load the images, process and write the result to out.png.

.. code-block:: python

   reconst = Reconstruction(wahrsis3, wahrsis4, datetime.datetime(2015,10,29,12,58,1))
   reconst.load_images(images3, images4)
   reconst.process()
   reconst.writeFigure("out.png")

Dependencies
------------

This project was built with python 2.7.12

Furthermore, the following packages were used:

=================  ========  =================================
Package            Version   Link
=================  ========  =================================
openCV             3.1.0     http://opencv.org/
numpy              1.12.0    http://www.numpy.org/
scipy              0.18.1    https://www.scipy.org/
scikit-image       0.12.3    http://scikit-image.org/
scikit-learn       0.18.1    http://scikit-learn.org/
matplotlib         2.0.0     http://matplotlib.org/
PyEphem            3.7.6.0   http://rhodesmill.org/pyephem/
pycuda (optional)  2016.1.2  https://documen.tician.de/pycuda/
=================  ========  =================================
