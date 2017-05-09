import cv2
import numpy as np
import datetime
import ephem
np.seterr(divide='ignore', invalid='ignore')
from collections import namedtuple

import colorsys
import matplotlib.colors
from sklearn.cluster import KMeans


class Segmentation():
	"""Contains all the methods needed for the segmentation a sky/cloud image
	
	Attributes
	----------
	image :  np.array
		The image to segment
	imager : Imager
		The imager used to take the image
	time: datetime.datetime
		The time at which the image was captured (UTC+8)"""
		
	def __init__(self, image, imager, time):
		self.img = image
		self.imager = imager
		self.time = time

	
	def showasImage(self,input_matrix):
		"""Normalizes an input matrix to the range [0,255]. It is useful in displaying the matrix as an image.
		
		Args:
			input_matrix (numpy array): Input matrix that needs to be normalized.
			
		Returns:
			numpy array: Returns the normalized matrix."""
		
		return (input_matrix - np.amin(input_matrix))/(np.amax(input_matrix) - np.amin(input_matrix))*255
	
	
	
	def make_cluster_mask(self,input_matrix,mask_image):
		"""Clusters an input sky/cloud image to generate the binary image and the coverage ratio value.
		
		Args:
			input_matrix (numpy array): Input matrix that needs to be normalized.
			mask_image (numpy array): Mask to remove occlusions from the input image. This mask contains boolean values indicating the allowable pixels from an image.
			
		Returns:
			numpy array: Binary output image, where white denotes cloud pixels and black denotes sky pixels.
			float: Cloud coverage ratio in the input sky/cloud image.
			float: The first (out of two) cluster center.
			float: The second (out of two) cluster center."""
	
		[rows,cols]=mask_image.shape
		
		im_mask_flt = mask_image.flatten()
		find_loc = np.where(im_mask_flt==1)
		find_loc = list(find_loc)
		
		input_vector = input_matrix.flatten()
		
		input_select = input_vector[list(find_loc)]
		
		
		X = input_select.reshape(-1, 1)
		k_means = KMeans(init='k-means++', n_clusters=2)
		k_means.fit(X)
		k_means_labels = k_means.labels_
		k_means_cluster_centers = k_means.cluster_centers_
		k_means_labels_unique = np.unique(k_means_labels)
		
	
		center1 = k_means_cluster_centers[0]
		center2 = k_means_cluster_centers[1]
		
		
		if center1 < center2:
			# Interchange the levels.
			temp = center1
			center1 = center2
			center2 = temp
	
			k_means_labels[k_means_labels == 0] = 99
			k_means_labels[k_means_labels == 1] = 0
			k_means_labels[k_means_labels == 99] = 1
			
		
		cent_diff = np.abs(center1-center2)
		if cent_diff<20:
			# Segmentation not necessary.
			if center1>120 and center2>120:
				# All cloud image
				k_means_labels[k_means_labels == 0] = 1
			else:
				# All sky image
				k_means_labels[k_means_labels == 1] = 0
			
		
		# 0 is sky and 1 is cloud
		cloud_pixels = np.count_nonzero(k_means_labels == 1)
		sky_pixels = np.count_nonzero(k_means_labels == 0)
		total_pixels = cloud_pixels + sky_pixels
		
		# print (cloud_pixels,total_pixels)
		cloud_coverage = float(cloud_pixels)/float(total_pixels)
		
		# Final threshold image for transfer
		index = 0
		Th_image = np.zeros([rows,cols])
		for i in range(0,rows):
			for j in range(0,cols):
				
				if mask_image[i,j]==1:
					#print (i,j)
					#print (index)
					Th_image[i,j] = k_means_labels[index]
					index = index + 1

		return(Th_image,cloud_coverage,center1,center2)
	
	
		
	def getBRchannel(self, input_image, mask_image):
		"""Extracts the ratio of red and blue blue channel from an input sky/cloud image. It is used in the clustering step to generate the binary sky/cloud image.
		
		Args:
			input_image (numpy array): Input sky/cloud image.
			mask_image (numpy array): Mask to remove occlusions from the input image. This mask contains boolean values indicating the allowable pixels from an image.
			
		Returns:
			numpy array: Ratio image using red and blue color channels, normalized to [0,255]."""
		
		red = input_image[:,:,2]
		green = input_image[:,:,1]
		blue = input_image[:,:,0]
		
		# RGB images for transfer
		red_image = red.astype(float) * mask_image
		green_image = green.astype(float) * mask_image
		blue_image = blue.astype(float) * mask_image
		
		BR = (blue_image - red_image) / (blue_image + red_image)
		BR[np.isnan(BR)] = 0
		
		return self.showasImage(BR)
	
	
	
	def cmask(self,index,radius,array):
		"""Generates the mask for a given input image. The generated mask is needed to remove occlusions during post-processing steps.
		
		Args:
			index (numpy array): Array containing the x- and y- co-ordinate of the center of the circular mask.
			radius (float): Radius of the circular mask.
			array (numpy array): Input sky/cloud image for which the mask is generated.

		Returns:
			numpy array: Generated mask image."""
  	
		a,b = index
		is_rgb = len(array.shape)
		
		if is_rgb == 3:
				ash = array.shape
				nx=ash[0]
				ny=ash[1]
		else:
				nx,ny = array.shape

		s = (nx,ny)
		image_mask = np.zeros(s)
		y,x = np.ogrid[-a:nx-a,-b:ny-b]
		mask = x*x + y*y <= radius*radius
		image_mask[mask] = 1
		
		return(image_mask)
	  
	  
	def removeSunIfNonCloudy(self, im_mask):
		"""Removes the circumsolar area from the mask if it is detected as non cloudy.
		
		Args:
			im_mask (numpy array): Segmentation mask

		Returns:
			numpy array: Modified segmentation mask."""
		
		
		if not self.imager.longitude or not self.imager.latitude or not self.imager.elevation:
			raise RuntimeError("Plese provide longitude, latitude and elevation for imager " + self.imager.name)
		
		# Detecting the sun location and checking if covered by clouds
		obs = ephem.Observer()
		obs.lon = self.imager.longitude
		obs.lat = self.imager.latitude
		obs.elevation = self.imager.elevation
		obs.date = self.time - datetime.timedelta(hours=8)
		sun = ephem.Sun(obs)

		raySun = np.array([[np.cos(sun.alt) * np.cos(sun.az), np.cos(sun.alt) * np.sin(sun.az), -np.sin(sun.alt)]])
		raySun = (np.dot(self.imager.rot.T, raySun.T - self.imager.trans)).T
		coordsSun = self.imager.world2cam(raySun.T)
		lum = 0.2126 * self.img[:,:,2].astype('float') + 0.7152 * self.img[:,:,1].astype('float') + 0.0722 * self.img[:,:,0].astype('float')
		maskSun = np.zeros(np.shape(lum))
		maskSunCircum = np.zeros(np.shape(lum))
		cv2.circle(maskSun, (coordsSun[1], coordsSun[0]), 100, (255), -1)
		cv2.circle(maskSunCircum, (coordsSun[1], coordsSun[0]), 250, (255), -1)
		maskCircum = maskSunCircum - maskSun
		lumSun = np.mean(lum[maskSun == 255])
		lumCircum = np.mean(lum[maskCircum == 255])

		if lumSun > 200 and lumSun - lumCircum > 30:
			maskSunCircum = cv2.resize(maskSunCircum, (np.shape(im_mask)[1], np.shape(im_mask)[0]))
			im_mask[maskSunCircum == 255] = 0
			
		return im_mask
		
		
	def segment(self):
		"""Main function for sky/cloud segmentation. Perform all the necessary steps."""
		
		# Segment the image
		im_mask = self.cmask(self.imager.center, self.imager.radius, self.img)
		im_mask = cv2.resize(im_mask, None, fx=0.25, fy=0.25)
		
	
		# Extract the color channels
		medR = cv2.resize(self.img, None, fx=0.25, fy=0.25)
		#(cc) = self.color16mask(medR, im_mask)
		
		self.removeSunIfNonCloudy(im_mask)
		
		inp_mat = self.getBRchannel(medR, im_mask)
		inp_mat[np.isnan(inp_mat)] = 0
		(th_img,coverage,center1, center2) = self.make_cluster_mask(inp_mat,im_mask)
	    
		cent_diff = np.abs(center1-center2)
		if cent_diff < 18:
			# Segmentation not necessary.
			if center1 > 120 and center2 > 120:
				# All cloud image
				self.th_img = np.zeros(np.shape(im_mask))
			else:
				# All sky image
				self.th_img = np.ones(np.shape(im_mask))
		else:
			self.th_img = inp_mat < center1 + (center2 - center1)/2
		self.th_img[im_mask == 0] = 0
		
		return self.th_img
		
		
