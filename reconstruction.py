import numpy as np
import itertools
import cv2
import skimage.exposure
import matplotlib.tri
import matplotlib.pyplot as plt
import tempfile
import os
import timeit
from mpl_toolkits.mplot3d import Axes3D
from segmentation import Segmentation



class Reconstruction():
	"""Contains all the methods for creating a 3D reconstruction from the base height from two sky pictures taken at the same time. Most functions need to be applied in the correct order to complete the workflow. See main.py for an example.
	
	Attributes
	----------
	imager1 : Imager
		Imager used to take the first image
	imager2 : Imager
		Imager used to take the second image
	time: datetime.datetime
		Time at which the images were captured
	sizeUndistorted: list[int]
		Size of the undistorted planes
	altitudes: list[int]
		Altitudes of the undistorted planes
	pixelWidths: list[int]
		Widths in meters of one pixel of the undistorted plane
	tilts: list[list[int]
		List of angles [elevation, azimuth] of the undistorted planes
	images: dict[list[numpy.array]]
		Dictionary (over each imager) of arrays of LDR images captured by that imager
	undistorted: dict[list[numpy.array]]
		Dictionary of list of LDR undistorted planes for each combination of undistortion parameters for the first imager
	tonemapped: dict[dict[numpy.array]]
		Dictionary (over each imager) of dictionaries of tonemapped images for each combination of undistortion parameters for that imager
	pts: dict[dict[numpy.array]]
		Dictionary (over each imager) of dictionaries of n*2 feature points locations in the undistorted planes of the that imager
	rays: dict[dict[numpy.array]]
		Dictionary (over each imager) of dictionary of 3*n vectors from the origin at the center of the imager towards each feature point in the planes for the that imager
	ptsWorld: dict[numpy.array]
		Dictionary of 3*n' coordinates of the reconstructed points, at the midpoint of the shortest segment between both rays from both imagers
	ptsWorldMerged: numpy.array
		ptsWorld concatenated in a single array independently of undistortion planes
	ptsWorldRays: dict[dict[numpy.array]]
		Dictionary (over each imager) of dictionary of 3*n' coordinates of the reconstructed points, on the rays originating from the first imager
	ptsWorldRaysMerged: dict[dict[numpy.array]]
		Dictionary (over each imager) of the reconstructed points in ptsWorldRays (all undistortion planes merged in common arrays)
	ptsPictureImager: dict[dict[numpy.array]]
		Dictionary (over each imager) of dictionaries of 2*n' vectors of the image plane locations corresponding to each reconstructed points for that imager
	ptsPictureImagerMerged: dict[numpy.array]
		Dictionary (over each imager) of the image locations in ptsPictureImager (all undistortion planes merged in common arrays)
	distances: dict[numpy.array]
		Dictionary of n' vectors of the lenghts of the shortest segments between the rays of the first and the second imager
	distancesMerged: numpy.array
		distances concatenated in a single array independently of undistortion planes
	triangulationError: float
		Mean lenghth of the shortest segments between rays for each pair of matching features
	meanHeight: float
		Mean z coordinate of the reconstructed points
	stdHeight: float
		Standard deviation of the z coordinates of the reconstructed points
	nbPts: int
		Number of reconstructed points
	cloudBaseHeight: float
		10th percentile of the z coordinates of the reconstructed points, assumed to be the cloud base height
	segmentationMask: np.array
		Sky/cloud segmentation mask for the first imager
	triangulation: matplotlib.tri.Triangulation
		Delaunay triangulation between the reconstructed points"""
		
		
	def __init__(self, imager1, imager2, time, sizeUndistorted = [1000, 1000, 3], altitudes = [500], pixelWidths = [1], tilts = [[0, 0], [35, 0], [35, 90], [35, 180], [35, 270]]):
		self.imager1 = imager1
		self.imager2 = imager2
		self.time = time
		self.sizeUndistorted = sizeUndistorted
		self.altitudes = altitudes
		self.pixelWidths = pixelWidths
		self.tilts = tilts
		
		self.images = {}
		self.undistorted = {}
		self.tonemapped = {}
		self.pts = {}
		self.rays = {}
		self.ptsWorld = {}
		self.ptsWorldMerged = None
		self.ptsWorldRays = {}
		self.ptsWorldRaysMerged = {}
		self.ptsPictureImager = {}
		self.ptsPictureImagerMerged = {}
		self.distances = {}
		self.distancesMerged = None
		self.triangulationError = None
		self.meanHeight = None
		self.stdHeight = None
		self.nbPts = None
		self.cloudBaseHeight = None
		self.segmentationMask = None
		self.triangulation = None


	def load_images(self, images1, images2):
		"""Loads the input LDR images captured at the same instant by both imagers.
		
		Args:
			images1 (list[numpy.array]): LDR images captured by imager1
			images2 (list[numpy.array]): LDR images captured by imager2"""
		
		self.images[self.imager1] = images1
		self.images[self.imager2] = images2



	def process(self):
		"""Executes the entire reconstruction process. Alternatively, every function can be called individually. :func:`~reconstruction.Reconstruction.load_images` must be called before."""
		
		if not self.images:
			raise RuntimeError("load_images() must be called first")
		
		self.undistort()
		self.tonemap()
		self.match()
		self.computeRays()
		self.intersectRays()
		self.removeOutliers()
		self.concatenateValues()
		self.computeMetrics()
		self.computeTriangulation()



	def iter_planes(self):
		"""Helper function to iterate over all undistortion plane parameters (altitudes, pixel width and tilts).
		
		Returns:
			iterator: over all (altitude, pixel width, tilts) tuples"""
		
		return itertools.product(self.altitudes, self.pixelWidths, self.tilts)



	def undistort(self):
		"""Generates the undistortion planes. :func:`~reconstruction.Reconstruction.load_images` must be called before."""
		
		start = timeit.default_timer()
		print "Undistorting images",
		
		if not self.images:
			raise RuntimeError("load_images() must be called first")
		
		# Iterate over both imagers
		for imager in [self.imager1, self.imager2]:
			self.undistorted[imager] = {}
			
			# Iterate over all set of undistortion parameters
			for (altitude, pixelWidth, tilt) in self.iter_planes():
				
				# Store the images in the dictionary using a unique key
				key = str(altitude) + '-' + str(pixelWidth) + '-' + str(tilt[0]) + '-' + str(tilt[1])
				
				self.undistorted[imager][key] = []
				for img in self.images[imager]:
					self.undistorted[imager][key].append(imager.undistort_image(img, self.sizeUndistorted, altitude, tilt[1], tilt[0], pixelWidth))

		stop = timeit.default_timer()
		print ': ' + str(stop - start)



	def tonemap(self):
		"""Tonemapps the undistorted planes. :func:`~reconstruction.Reconstruction.undistort` must be called before."""
		
		start = timeit.default_timer()
		print "Tonemapping undistortion planes",
		
		if not self.undistorted:
			raise RuntimeError("undistort() must be called first")
		
		# HDR fusion using Mertens method
		merge_mertens = cv2.createMergeMertens()
		
		# Iterate over both imagers
		for imager in [self.imager1, self.imager2]:
			self.tonemapped[imager] = {}
			
			# Iterate over all set of undistortion parameters
			for (altitude, pixelWidth, tilt) in self.iter_planes():
				key = str(altitude) + '-' + str(pixelWidth) + '-' + str(tilt[0]) + '-' + str(tilt[1])
	
				# HDR fusion, rescaling and tonemapping
				tm = merge_mertens.process(self.undistorted[imager][key])
				tm = skimage.exposure.rescale_intensity(tm, in_range=(0,1), out_range='uint16').astype('uint16')
				
				# Tonemapping using CLAHE (Contrast Limited Adaptive Histogram Equalization)
				tm = skimage.exposure.equalize_adapthist(tm)
				self.tonemapped[imager][key] = np.clip(255 * tm, 0, 255).astype('uint8')
			
		stop = timeit.default_timer()
		print ': ' + str(stop - start) 


	def match(self):
		"""Creates sift matching reconstruction between a pair of undistorted images. :func:`~reconstruction.Reconstruction.tonemap` must be called before."""
		
		start = timeit.default_timer()
		print "Computing matches",
		
		def SIFTmatching(im1, im2):
			"""Creates sift matching between two images."""
	 
			sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.06, edgeThreshold=150)
			k1, des1 = sift.detectAndCompute(im1, None)
			k2, des2 = sift.detectAndCompute(im2, None)
	 
			if len(k1) < 2 or len(k2) < 2:
				return [None, None]
	 
			# create brute force matcher object
			bf = cv2.BFMatcher(crossCheck=True)
	
			# Match descriptors
			matches = bf.match(np.array(des1), np.array(des2))
	
			# Sort them in the order of their distance
			matches = sorted(matches, key = lambda x:x.distance)
	 
			# store all the good matches as per Lowe's ratio test
			good = []
			for m in matches:
				if m.distance < 300:
					good.append(m)
			matches = good
	 
	 		pts1 = []
	 		pts2 = []
	 
	 		for m in matches:
				pts1.append([int(k1[m.queryIdx].pt[1]), int(k1[m.queryIdx].pt[0])])
				pts2.append([int(k2[m.trainIdx].pt[1]), int(k2[m.trainIdx].pt[0])])
	 
	 		return [np.array(pts1), np.array(pts2)]

		if not self.tonemapped:
			raise RuntimeError("tonemap() must be called first")

		self.pts[self.imager1] = {}
		self.pts[self.imager2] = {}

		# Iterate over all set of undistortion parameters
		for (altitude, pixelWidth, tilt) in self.iter_planes():
			key = str(altitude) + '-' + str(pixelWidth) + '-' + str(tilt[0]) + '-' + str(tilt[1])

			[self.pts[self.imager1][key], self.pts[self.imager2][key]] = SIFTmatching(self.tonemapped[self.imager1][key], self.tonemapped[self.imager2][key])
		
		stop = timeit.default_timer()
		print ': ' + str(stop - start)



	def computeRays(self):
		"""Computes the light rays from the pairs of matching points :func:`~reconstruction.Reconstruction.match` must be called before."""
		
		start = timeit.default_timer()
		print "Computing light ray directions",
		
		if not self.pts:
			raise RuntimeError("match() must be called first")		
		
		# Output coordinate system: East, North, Up
		
		# Iterate over both imagers
		for imager in [self.imager1, self.imager2]:
			self.rays[imager] = {}	
			
			# Iterate over all set of undistortion parameters
			for (altitude, pixelWidth, tilt) in self.iter_planes():
				key = str(altitude) + '-' + str(pixelWidth) + '-' + str(tilt[0]) + '-' + str(tilt[1])
				bearing = tilt[1]
				elev = tilt[0]
	         
	         # If there are points in this plane
				if self.pts[imager][key] is not None and len(self.pts[imager][key]):
					
					# Compute the rotations to apply to the plane
					bearingRad = np.radians(bearing)
					elevRad = np.radians(elev)
					rotElev = np.array([[1, 0, 0], [0, np.cos(elevRad), -np.sin(elevRad)], [0, np.sin(elevRad), np.cos(elevRad)]])
					rotBearing = np.array([[np.cos(bearingRad), -np.sin(bearingRad), 0], [np.sin(bearingRad), np.cos(bearingRad), 0], [0, 0, 1]])
	            
	            # Convert location of the points in the image to 3D coordinates with origin at the imager location
					coords = np.array([[pt[1], pt[0]] for pt in self.pts[imager][key]])
					coords[:,0] = pixelWidth * (coords[:,0] - self.sizeUndistorted[0]/2)
					coords[:,1] = pixelWidth * (coords[:,1] - self.sizeUndistorted[1]/2)
	
					coordsZ = -altitude * np.ones(coords[:,0].shape)
					coordsZ = coordsZ.reshape(np.shape(coordsZ)[0], 1)
					coords = np.hstack((coords, coordsZ))
	
					coords = np.dot(rotElev.T, coords.T)
					coords = np.dot(rotBearing.T, coords)
	
					coords[2,:] = -coords[2,:]
					self.rays[imager][key] = coords
				else:
					self.rays[imager][key] = None
				
		stop = timeit.default_timer()
		print ': ' + str(stop - start)



	def intersectRays(self, threshold = 100):
		"""Computes the intersection the light rays from the matching pairs :func:`~reconstruction.Reconstruction.computeRays` must be called before.
		
		Args:
			threshold (int): the minimum distance between two rays to discard a match."""
		
		if not self.rays:
			raise RuntimeError("computeRays() must be called first")
			
		start = timeit.default_timer()
		print "Intersecting light rays",
		
		self.ptsWorldRays[self.imager1] = {}
		self.ptsWorldRays[self.imager2] = {}
		self.ptsPictureImager[self.imager1] = {}
		self.ptsPictureImager[self.imager2] = {}

		coordsImg1 = self.imager1.position
		coordsImg2 = self.imager2.position
		b = coordsImg2 - coordsImg1
    
      # Iterate over all set of undistortion parameters
		for (altitude, pixelWidth, tilt) in self.iter_planes():
			key = str(altitude) + '-' + str(pixelWidth) + '-' + str(tilt[0]) + '-' + str(tilt[1])
         
			self.ptsWorld[key] = []
			self.ptsWorldRays[self.imager1][key] = []
			self.ptsWorldRays[self.imager2][key] = []
			self.distances[key] = []
			self.ptsPictureImager[self.imager1][key] = []
			self.ptsPictureImager[self.imager2][key] = []
         
			if self.rays[self.imager1][key] is not None and self.rays[self.imager2][key] is not None:
				# Finding the point the closest to both rays
				# Source: http://www.mathworks.com/matlabcentral/newsreader/view_thread/149005
				for idx in np.arange(np.shape(self.rays[self.imager1][key])[1]):
					a = np.array([self.rays[self.imager1][key][:,idx], -self.rays[self.imager2][key][:,idx]])
					x = np.linalg.lstsq(np.transpose(a), b)
					
					dist = np.linalg.norm(coordsImg1 + x[0][0] * self.rays[self.imager1][key][:,idx] - coordsImg2 - x[0][1] * self.rays[self.imager2][key][:,idx])
					if dist < threshold:
						self.distances[key].append(dist)
						thisPtsRay1 = coordsImg1 + x[0][0] * self.rays[self.imager1][key][:,idx]
						thisPtsRay2 = coordsImg2 + x[0][1] * self.rays[self.imager2][key][:,idx]
						self.ptsWorld[key].append((thisPtsRay1 + thisPtsRay2)/2)
						self.ptsWorldRays[self.imager1][key].append(thisPtsRay1)
						self.ptsWorldRays[self.imager2][key].append(thisPtsRay2)

						self.ptsPictureImager[self.imager1][key].append([self.pts[self.imager1][key][idx][1], self.pts[self.imager1][key][idx][0]])
						self.ptsPictureImager[self.imager2][key].append([self.pts[self.imager2][key][idx][1], self.pts[self.imager2][key][idx][0]])
						
		stop = timeit.default_timer()
		print ': ' + str(stop - start)



	def removeOutliers(self):
		"""Removes outliers in the set of reconstructed points :func:`~reconstruction.Reconstruction.intersectRays` must be called before."""
		
		start = timeit.default_timer()
		print "Removing outliers",
		
		if not self.ptsWorld:
			raise RuntimeError("intersectRays() must be called first")

		# Iterate over all set of undistortion parameters
		for (altitude, pixelWidth, tilt) in self.iter_planes():
			key = str(altitude) + '-' + str(pixelWidth) + '-' + str(tilt[0]) + '-' + str(tilt[1])

			if self.ptsWorld[key]:
				# Computing mean and standard deviation over the z coordinates of the reconstructed points
				Zmean = np.mean(np.array(self.ptsWorld[key])[:,2])
				Zstd = np.std(np.array(self.ptsWorld[key])[:,2])

				idxToKeep = []
				for idx, mid in enumerate(self.ptsWorld[key]):
					# Removing points beyond 1.5*std and below 300 meters or above 10,000 meters
					if not mid[2] < Zmean - 1.5*Zstd and not mid[2] > Zmean + 1.5*Zstd and mid[2] > 300 and mid[2] < 10000:
						idxToKeep.append(idx)

				# Removing the points in all the arrays
				self.ptsWorld[key] = [ self.ptsWorld[key][i] for i in idxToKeep ]
				self.distances[key] = [ self.distances[key][i] for i in idxToKeep ]
				self.ptsWorldRays[self.imager1][key] = [ self.ptsWorldRays[self.imager1][key][i] for i in idxToKeep ]
				self.ptsWorldRays[self.imager2][key] = [ self.ptsWorldRays[self.imager2][key][i] for i in idxToKeep ]
				self.ptsPictureImager[self.imager1][key] = [ self.ptsPictureImager[self.imager1][key][i] for i in idxToKeep ]
				self.ptsPictureImager[self.imager2][key] = [ self.ptsPictureImager[self.imager2][key][i] for i in idxToKeep ]
				
		stop = timeit.default_timer()
		print ': ' + str(stop - start)



	def concatenateValues(self):
		"""Concatenates all the computed values from all the different planes into a single vectors. :func:`~reconstruction.Reconstruction.intersectRays` must be called before."""
		
		start = timeit.default_timer()
		print "Merging values from all undistortion planes",
		
		self.ptsWorldMerged = np.array(list(itertools.chain.from_iterable(self.ptsWorld.values())))
		self.ptsPictureImagerMerged[self.imager1] = np.array(list(itertools.chain.from_iterable(self.ptsPictureImager[self.imager1].values())))
		self.ptsPictureImagerMerged[self.imager2] = np.array(list(itertools.chain.from_iterable(self.ptsPictureImager[self.imager2].values())))
		self.distancesMerged = np.array(list(itertools.chain.from_iterable(self.distances.values())))
		self.ptsWorldRaysMerged[self.imager1] = np.array(list(itertools.chain.from_iterable(self.ptsWorldRays[self.imager1].values())))
		self.ptsWorldRaysMerged[self.imager2] = np.array(list(itertools.chain.from_iterable(self.ptsWorldRays[self.imager2].values())))
		
		stop = timeit.default_timer()
		print ': ' + str(stop - start)



	def computeMetrics(self):
		"""Computes the light rays from the pairs of matching points :func:`~reconstruction.Reconstruction.concatenateValues` must be called before."""
		
		start = timeit.default_timer()
		print "Computing quality metrics",
		
		self.triangulationError = np.mean(self.distancesMerged)
		self.meanHeight = np.mean(np.array(self.ptsWorldMerged)[:,2])
		self.stdHeight = np.std(np.array(self.ptsWorldMerged)[:,2])
		self.nbPts = len(self.distancesMerged)
		# Cloud base height is assumed to be the 10th percentile of the z coordinates
		try:
			self.cloudBaseHeight = np.percentile(np.array(self.ptsWorldMerged)[:,2], 5)
		except:
			self.cloudBaseHeight = np.nan
			
		stop = timeit.default_timer()
		print ': ' + str(stop - start)



	def computeTriangulation(self):
		"""Creates the triangulation between reconstructed points for visualization. :func:`~reconstruction.Reconstruction.concatenateValues` must be called before."""
		
		start = timeit.default_timer()
		print "Computing Delaunay triangulation",
	    
		med = self.images[self.imager1][len(self.images[self.imager1])/2]
	   
		seg = Segmentation(med, self.imager1, self.time)
		self.segmentationMask = seg.segment()
		medR = cv2.resize(med, None, fx=0.25, fy=0.25)
	
		def triangleMask(v1x, v1y, v2x, v2y, v3x, v3y):
			b1 = ((xv - v2x) * (v1y - v2y) - (v1x - v2x) * (yv - v2y)) < 0.0
			b2 = ((xv - v3x) * (v2y - v3y) - (v2x - v3x) * (yv - v3y)) < 0.0
			b3 = ((xv - v1x) * (v3y - v1y) - (v3x - v1x) * (yv - v1y)) < 0.0
			return np.logical_and((b1 == b2), (b2 == b3))
	   
		if self.ptsWorldMerged.size:
	        
			world = np.array(self.ptsWorldMerged)
			world[:,2] = -world[:,2]
			world = (np.dot(self.imager1.rot, world.T) + self.imager1.trans).T
			world = world[:,[1,0,2]]
			img = self.imager1.world2cam(world.T).T[:,[1,0]]
	   
		if img.size and len(np.unique(img[:,0])) >= 3:

			self.triangulation = matplotlib.tri.Triangulation(img[:,0], img[:,1])
			mArray = []
	        
			xv, yv = np.meshgrid(np.arange(np.shape(medR)[1]), np.arange(np.shape(medR)[0]))
	      
			for t in self.triangulation.triangles:
				mask = triangleMask(img[t[0],0]/4, img[t[0],1]/4, img[t[1],0]/4, img[t[1],1]/4, img[t[2],0]/4, img[t[2],1]/4)
				realPoint1 = self.ptsWorldMerged[t[0],:]
				realPoint2 = self.ptsWorldMerged[t[1],:]
				realPoint3 = self.ptsWorldMerged[t[2],:]

				a = np.linalg.norm(realPoint1 - realPoint2)
				b = np.linalg.norm(realPoint2 - realPoint3)
				c = np.linalg.norm(realPoint3 - realPoint1)
				s = 0.5 * (a + b + c)
				A = np.sqrt(s * (s-a) * (s-b) * (s-c))
	         
				if not (np.isnan(self.segmentationMask[mask])).all():
					ratio = np.mean(self.segmentationMask[mask])
				else:
					ratio = 0
					
				if ratio > 0.5 and A < 200000:
					mArray.append(False)
				else:
					mArray.append(True)
	            
			self.triangulation.set_mask(mArray)
			
		stop = timeit.default_timer()
		print ': ' + str(stop - start)



	def writeFigure(self, filename):
		"""Creates a visualization of the reconstructed elements. :func:`~reconstruction.Reconstruction.createTriangulation` must be called before."""
		
		start = timeit.default_timer()
		print "Generating output figure",
	    
		fig = plt.figure(frameon = False)
		fig.set_size_inches(15,10)
		ax1 = fig.add_subplot(131, projection='3d')
		ax2 = fig.add_subplot(132, projection='3d')
		ax3 = fig.add_subplot(133, projection='3d')
		plt.tight_layout()
	    
		ax1.set_aspect('equal')
		ax1.set_xlabel('W - E')
		ax1.set_ylabel('S - N')
		ax1.set_zlabel('Height')
		ax1.set_xlim([-2000, 2000])
		ax1.set_ylim([-2000, 2000])
		ax1.set_zlim(0, 2000)
	    
		ax2.set_aspect('equal')
		ax2.set_xlabel('W - E')
		ax2.set_zlabel('Height')
		plt.setp( ax2.get_yticklabels(), visible=False)
		ax2.set_xticks([-2000,-1000,0,1000,2000])
		ax2.set_xlim([-2000, 2000])
		ax2.set_ylim([-2000, 2000])
		ax2.set_zlim(0, 2000)
	    
		ax3.set_aspect('equal')
		ax3.set_xlabel('W - E')
		ax3.set_ylabel('S - N')
		plt.setp( ax3.get_zticklabels(), visible=False)
		ax3.set_xticks([-2000,-1000,0,1000,2000])
		ax3.set_xlim([-2000, 2000])
		ax3.set_ylim([-2000, 2000])
		ax3.set_zlim(0, 2000)
	
		color = 'black'
	
		xs = [mid[0] for mid in self.ptsWorldMerged]
		ys = [mid[1] for mid in self.ptsWorldMerged]
		zs = [mid[2] for mid in self.ptsWorldMerged]
		ax1.scatter(xs, ys, zs, c=color, s=10, lw = 0)
		ax2.scatter(xs, ys, zs, c=color, s=10, lw = 0)
		ax3.scatter(xs, ys, zs, c=color, s=10, lw = 0)
	
		if self.triangulation:
			for t in self.triangulation.get_masked_triangles():
				t03d = self.ptsWorldMerged[t[0]]
				t13d = self.ptsWorldMerged[t[1]]
				t23d = self.ptsWorldMerged[t[2]]

				ax1.plot([t03d[0], t13d[0], t23d[0], t03d[0]], [t03d[1], t13d[1], t23d[1], t03d[1]], [t03d[2], t13d[2], t23d[2], t03d[2]], color)
				ax2.plot([t03d[0], t13d[0], t23d[0], t03d[0]], [t03d[1], t13d[1], t23d[1], t03d[1]], [t03d[2], t13d[2], t23d[2], t03d[2]], color)
				ax3.plot([t03d[0], t13d[0], t23d[0], t03d[0]], [t03d[1], t13d[1], t23d[1], t03d[1]], [t03d[2], t13d[2], t23d[2], t03d[2]], color)
	
		ax1.scatter(self.imager1.position[0], self.imager1.position[1], self.imager1.position[2], c=(0,1,0), s=50)
		ax1.scatter(self.imager2.position[0], self.imager2.position[1], self.imager2.position[2], c=(1,0,0), s=50)
	    
		ax2.scatter(self.imager1.position[0], self.imager1.position[1], self.imager1.position[2], c=(0,1,0), s=50)
		ax2.scatter(self.imager2.position[0], self.imager2.position[1], self.imager2.position[2], c=(1,0,0), s=50)
	    
		ax3.scatter(self.imager1.position[0], self.imager1.position[1], self.imager1.position[2], c=(0,1,0), s=50)
		ax3.scatter(self.imager2.position[0], self.imager2.position[1], self.imager2.position[2], c=(1,0,0), s=50)
	    
		ax1.view_init(elev=30, azim=-60)
		ax2.view_init(elev=0, azim=-90)
		ax3.view_init(elev=-90, azim=-90)
	   
		tmpfile = os.path.join(tempfile._get_default_tempdir(), 'out.png')
		plt.savefig(tmpfile, bbox_inches = 'tight')
		plt.clf()
		ax = plt.Axes(fig, [0., 0., 1., 1.], )
		ax.set_axis_off()
		fig.add_axes(ax)
		plots3D = cv2.imread(tmpfile)
		
		fisheye1 = self.images[self.imager1][len(self.images[self.imager1])/2]
		fisheye1 = cv2.resize(fisheye1, None, fx=0.25, fy=0.25)
		fisheye1[:,:,2] = self.segmentationMask * fisheye1[:,:,0]
		fisheye1[:,:,1] = self.segmentationMask * fisheye1[:,:,1]
		fisheye1[:,:,0] = self.segmentationMask * fisheye1[:,:,2]
		ax.imshow(fisheye1, aspect='auto')
	   
		worldCalib = np.array(self.ptsWorldMerged)
		if worldCalib.size:
			dists = np.array(self.distancesMerged)
			worldCalib[:,2] = -worldCalib[:,2]
			worldCalib = (np.dot(self.imager1.rot, worldCalib.T) + self.imager1.trans).T
			worldCalib = worldCalib[:,[1,0,2]]
			img = self.imager1.world2cam(worldCalib.T).T
			img[:,0] = img[:,0]/4.0
			img[:,1] = img[:,1]/4.0

			#ax.scatter(img[:,1], img[:,0], c=dists, s=30, lw=0, zorder=5)
			ax.scatter(img[:,1], img[:,0], c='green', s=30, lw=0, zorder=5)
			if self.triangulation:
				for t in self.triangulation.get_masked_triangles():
					t03d = img[t[0],:]
					t13d = img[t[1],:]
					t23d = img[t[2],:]

					ax.plot([t03d[1], t13d[1], t23d[1], t03d[1]], [t03d[0], t13d[0], t23d[0], t03d[0]], 'red', linewidth=1.0, zorder=3)
	   
		fig.savefig(tmpfile, bbox_inches='tight')
		plt.clf()
		fisheye1 = cv2.imread(tmpfile)
		# Removing white borders
		fisheye1 = fisheye1[np.where(np.mean(fisheye1[:,:,0], axis=1) < 250), :, :].squeeze()
		fisheye1 = fisheye1[:, np.where(np.mean(fisheye1[:,:,0], axis=0) < 250), :].squeeze()

		fisheye2 = self.images[self.imager2][len(self.images[self.imager2])/2]
		triangulationError = self.triangulationError
		meanHeight = self.meanHeight
		stdHeight = self.stdHeight
		cbHeight = self.cloudBaseHeight
		nbPts = self.nbPts

		sFisheye = (333, 500)
		s3D = np.shape(plots3D)
		
		fisheye1 = cv2.resize(fisheye1, (sFisheye[1], sFisheye[0]))
		fisheye2 = cv2.resize(fisheye2, (sFisheye[1], sFisheye[0]))

		outImage = 255*np.ones((sFisheye[0] + s3D[0], max([2*sFisheye[1], s3D[1]]), 3)).astype('uint8')
		outImage[:sFisheye[0], :, :] = 0
		outImage[-s3D[0]:, :s3D[1], :] = plots3D
		outImage[:sFisheye[0], :sFisheye[1], :] = fisheye1
		outImage[:sFisheye[0], -sFisheye[1]:, :] = fisheye2

		sOut = np.shape(outImage)
		cv2.putText(outImage, self.time.strftime("%d-%m-%Y"), (int(sOut[1]/2)-50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
		cv2.putText(outImage, self.time.strftime("%H-%M-%S"), (int(sOut[1]/2)-65, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255))
		cv2.putText(outImage, "Triangulation error [m]:", (int(sOut[1]/2)-80, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
		cv2.putText(outImage, "%.2f" % triangulationError, (int(sOut[1]/2)-20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
		cv2.putText(outImage, "Mean height [m]:", (int(sOut[1]/2)-60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
		cv2.putText(outImage, "%.2f" % meanHeight, (int(sOut[1]/2)-30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
		cv2.putText(outImage, "Std height [m]:", (int(sOut[1]/2)-55, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
		cv2.putText(outImage, "%.2f" % stdHeight, (int(sOut[1]/2)-35, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
		cv2.putText(outImage, "Cloud base height [m]:", (int(sOut[1]/2)-80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
		cv2.putText(outImage, "%.2f" % cbHeight, (int(sOut[1]/2)-30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
		cv2.putText(outImage, "Number of points:", (int(sOut[1]/2)-70, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
		cv2.putText(outImage, str(nbPts), (int(sOut[1]/2)-15, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
	    
		cv2.imwrite(filename, outImage)
		
		stop = timeit.default_timer()
		print ': ' + str(stop - start)
		
		