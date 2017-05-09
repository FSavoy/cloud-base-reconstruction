from imager import Imager
from reconstruction import Reconstruction
import numpy as np
import cv2
import datetime

# Sample usage file

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

name4 = 'wahrsis4'
center4 = [2000, 2975]
radius4 = 1665
relativePosition4 = np.array([-2.334, 101.3731, -8.04])
calibRot4 = np.array([[0.9710936, -0.23401871, 0.04703662], [0.234924, 0.97190314, -0.01466276], [-0.04228367, 0.02528894, 0.99878553]])
calibTrans4 = np.array([[-0.00274625], [-0.00316865], [0.00516088]])
wahrsis4 = Imager(name4, center4, radius4, relativePosition4, calibRot4, calibTrans4)

images3 = [cv2.imread('wahrsis3/2015-10-29-12-58-01-wahrsis3-low.jpg'),
				cv2.imread('wahrsis3/2015-10-29-12-58-01-wahrsis3-med.jpg'),
				cv2.imread('wahrsis3/2015-10-29-12-58-01-wahrsis3-high.jpg')]
images4 = [cv2.imread('wahrsis4/2015-10-29-12-58-01-wahrsis4-low.jpg'),
				cv2.imread('wahrsis4/2015-10-29-12-58-01-wahrsis4-med.jpg'),
				cv2.imread('wahrsis4/2015-10-29-12-58-01-wahrsis4-high.jpg')]

reconst = Reconstruction(wahrsis3, wahrsis4, datetime.datetime(2015,10,29,12,58,1))
reconst.load_images(images3, images4)
reconst.process()
reconst.writeFigure("out.png")