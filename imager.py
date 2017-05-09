import numpy as np

try:
	import pycuda.autoinit
	import pycuda.driver as drv
	from pycuda.compiler import SourceModule
	cuda = True
except Exception as e:
	print "Cuda disabled"
	print e
	cuda = False
	
from scipy import interpolate

class Imager:
	"""Describes an imager and all the parameters related to it. Contains the methods dealing with lens distortions and mis-alignment correction.
	
	Attributes
	----------
	name : str
		Name of the imager
	center : list[int]
		List of two ints describing the center of the fish-eye lens in the captured pictures
	radius: int
		The radius of the circle created by the fish-eye lens
	position: numpy.array
		The coordinates of the relative position of the imager in real world coordinates (meters)
	rot: numpy.array 
		A rotation matrix (3x3) to cope with the misalignement of the image
	trans: numpy.array
		A translation matrix (3x1) to cope with the misalignement of the image
	longitude: str
		The longitude coordinate of the imager (optional, only needed for segmentation)
	latitude: str
		The latitude coordinate of the imager (optional, only needed for segmentation)
	elevation: int
		The elevation of the integer in meters (optional, only needed for segmentation)"""
		
	def __init__(self, name, center, radius, position, rot, trans, longitude = None, latitude = None, elevation = None):
		self.name = name
		self.center = center
		self.radius = radius
		self.position = position
		self.rot = rot
		self.trans = trans
		self.longitude = longitude
		self.latitude = latitude
		self.elevation = elevation
		
		

	def __str__(self):
		"""Returns the name as the default string."""
		return self.name
		

	def cam2world(self, m):
		"""Returns the 3D coordinates with unit length corresponding to the pixels locations on the image.
		
		Args:
			m (numpy.array): 2 x m matrix containing m pixel coordinates
	
		Returns:
			numpy.array: 3 x m matrix of the corresponding unit sphere vectors"""
	    
		m1 = m[0,:] - self.center[0]
		m2 = m[1,:] - self.center[1]
		azimuths = np.arctan2(m2, m1)
	
		r = np.sqrt(np.power(m1,2) + np.power(m2,2))
		theta = np.pi/2 - 2*np.arcsin(r/(self.radius/np.sin(np.pi/4)))
	
		M1 = m1
		M2 = m2
		M3 = r*np.tan(-theta)
		norm = np.sqrt(np.power(M1, 2) + np.power(M2, 2) + np.power(M3, 2))
	
		return np.column_stack((M1/norm, M2/norm, M3/norm)).T


	def world2cam(self, M):
		"""Returns the pixel locations corresponding to the given 3D points.
		
		Args:
			M (numpy.array): 3 x m matrix containing M point coordinates
	
		Returns:
			numpy.array: 2 x m matrix of the corresponding pixel coordinates"""
	   
		azimuths = np.arctan2(M[0,:], M[1,:])
		elevations = np.arccos(-M[2,:]/np.sqrt(np.power(M[0,:],2) + np.power(M[1,:],2) + np.power(M[2,:],2)))
	
		r = self.radius/np.sin(np.pi/4)*np.sin(elevations/2)
		m1 = self.center[0] + r*np.sin(azimuths)
		m2 = self.center[1] + r*np.cos(azimuths)
	
		return np.column_stack((m1, m2)).T
		
		
	def undistort_image(self, image, sizeResult = [1000, 1000], altitude = 500, bearing = 0, elevation = 0, pixelWidth = 1):
		"""Undistort a fish-eye image to a standard camera image by a ray-tracing approach.
		
		Args:
			image (numpy.array): the fish-eye image to undistort
			sizeResult (list[int]): 2D size of the output undistorted image [pixels]
			altitude (int): the altitude of the virtual image plane to form the undistorted image [meters]
			bearing (float): the bearing angle of the undistorted plane (in degrees)
			elevation (float): the elevation angle of the undistorted plane (in degrees)
			pixelWidth (int): the width in real world coordinates of one pixel of the undistorted image [meters]"""
    
		centerImage = np.array([sizeResult[1]/2, sizeResult[0]/2])
	   
		if len(np.shape(image)) == 3:
			R = image[:,:,0]
			G = image[:,:,1]
			B = image[:,:,2]
	    
		sizeImg = image.shape
		posX, posY = np.meshgrid(np.arange(sizeImg[1]), np.arange(sizeImg[0]))
		posX = posX + 1
		posY = posY + 1
	
		mapY, mapX = np.meshgrid(np.arange(sizeResult[1]), np.arange(sizeResult[0]))
		mapX = mapX + 1
		mapY = mapY + 1
		distCenterX = (mapX.ravel(order='F') - centerImage[0]) * pixelWidth
		distCenterY = (mapY.ravel(order='F') - centerImage[1]) * pixelWidth
		distCenterZ = np.tile(-altitude, (distCenterX.shape[0], 1))
	    
		bearing = np.radians(bearing)
		elevation = np.radians(elevation)
		rotHg = np.array([[np.cos(elevation), 0, np.sin(elevation)], [0, 1, 0], [-np.sin(elevation), 0, np.cos(elevation)]])
		rotBearing = np.array([[np.cos(bearing), -np.sin(bearing), 0], [np.sin(bearing), np.cos(bearing), 0], [0, 0, 1]])
	    
		worldCoords = np.column_stack((distCenterX, distCenterY, distCenterZ)).T
	    
		worldCoords = np.dot(rotHg, worldCoords)
		worldCoords = np.dot(rotBearing, worldCoords)
		worldCoords = np.dot(self.rot.T, worldCoords - self.trans)
	   
		m = self.world2cam(worldCoords)
	
		if len(np.shape(image)) == 3:
			if cuda:
				result = self.cuda_interpolate3D(image, m, (sizeResult[0], sizeResult[1]))
			else:
				ip = interpolate.RectBivariateSpline(np.arange(sizeImg[0]), np.arange(sizeImg[1]), R)
				resultR = ip.ev(m[0,:]-1, m[1,:]-1)
				resultR = resultR.reshape(sizeResult[0], sizeResult[1],order='F')
				np.clip(resultR, 0, 255, out=resultR)
				resultR = resultR.astype('uint8')
	
				ip = interpolate.RectBivariateSpline(np.arange(sizeImg[0]), np.arange(sizeImg[1]), G)
				resultG = ip.ev(m[0,:]-1, m[1,:]-1)
				resultG = resultG.reshape(sizeResult[0], sizeResult[1],order='F')
				np.clip(resultG, 0, 255, out=resultG)
				resultG = resultG.astype('uint8')
	
				ip = interpolate.RectBivariateSpline(np.arange(sizeImg[0]), np.arange(sizeImg[1]), B)
				resultB = ip.ev(m[0,:]-1, m[1,:]-1)
				resultB = resultB.reshape(sizeResult[0], sizeResult[1],order='F')
				np.clip(resultB, 0, 255, out=resultB)
				resultB = resultB.astype('uint8')
	
				result = np.zeros(sizeResult).astype('uint8')
				result[:,:,0] = resultR
				result[:,:,1] = resultG
				result[:,:,2] = resultB
		else:
			if cuda:
				result = self.cuda_interpolate(image, m, (sizeResult[0], sizeResult[1]))
			else:
				ip = interpolate.RectBivariateSpline(np.arange(sizeImg[0]), np.arange(sizeImg[1]), image)
				result = ip.ev(m[0,:]-1, m[1,:]-1)
				result = result.reshape(sizeResult[0], sizeResult[1],order='F')
				np.clip(result, 0, 255, out=result)
				result = result.astype('uint8')
	    
		return result
		
		
	def cuda_interpolate(self, channel, m, size_result):
		cols = size_result[0]; rows = size_result[1];
	
		kernel_code = """
		texture<float, 2> tex;
	
		__global__ void interpolation(float *dest, float *m0, float *m1)
		{
			int idx = threadIdx.x + blockDim.x * blockIdx.x;
			int idy = threadIdx.y + blockDim.y * blockIdx.y;
	
			if (( idx < %(NCOLS)s ) && ( idy < %(NDIM)s )) {
				dest[%(NDIM)s * idx + idy] = tex2D(tex, m0[%(NDIM)s * idy + idx], m1[%(NDIM)s * idy + idx]);
			}
		}
		"""
	    
		kernel_code = kernel_code % {'NCOLS': cols, 'NDIM': rows}
		mod = SourceModule(kernel_code)
	
		interpolation = mod.get_function("interpolation")
		texref = mod.get_texref("tex")
	
		channel = channel.astype("float32")
		drv.matrix_to_texref(channel, texref, order="F")
		texref.set_filter_mode(drv.filter_mode.LINEAR)
	
		bdim = (16, 16, 1)
		dx, mx = divmod(cols, bdim[0])
		dy, my = divmod(rows, bdim[1])
	
		gdim = ((dx + (mx>0)) * bdim[0], (dy + (my>0)) * bdim[1])
	
		dest = np.zeros((rows,cols)).astype("float32")
		m0 = (m[0,:]-1).astype("float32")
		m1 = (m[1,:]-1).astype("float32")
	
		interpolation(drv.Out(dest), drv.In(m0), drv.In(m1), block=bdim, grid=gdim, texrefs=[texref])
	
		return dest.astype("uint8")
	
	
	def cuda_interpolate3D(self, img, m, size_result):
		cols = size_result[0]; rows = size_result[1];
	
		kernel_code = """
		texture<float, 2> texR;
		texture<float, 2> texG;
		texture<float, 2> texB;
	
		__global__ void interpolation(float *dest, float *m0, float *m1)
	    {
			int idx = threadIdx.x + blockDim.x * blockIdx.x;
			int idy = threadIdx.y + blockDim.y * blockIdx.y;
	
			if (( idx < %(NCOLS)s ) && ( idy < %(NDIM)s )) {
				dest[3*(%(NDIM)s * idx + idy)] = tex2D(texR, m0[%(NDIM)s * idy + idx], m1[%(NDIM)s * idy + idx]);
				dest[3*(%(NDIM)s * idx + idy) + 1] = tex2D(texG, m0[%(NDIM)s * idy + idx], m1[%(NDIM)s * idy + idx]);
				dest[3*(%(NDIM)s * idx + idy) + 2] = tex2D(texB, m0[%(NDIM)s * idy + idx], m1[%(NDIM)s * idy + idx]);
			}
		}
		"""
	    
		kernel_code = kernel_code % {'NCOLS': cols, 'NDIM': rows}
		mod = SourceModule(kernel_code)
	
		interpolation = mod.get_function("interpolation")
		texrefR = mod.get_texref("texR")
		texrefG = mod.get_texref("texG")
		texrefB = mod.get_texref("texB")
	
		img = img.astype("float32")
		drv.matrix_to_texref(img[:,:,0], texrefR, order="F")
		texrefR.set_filter_mode(drv.filter_mode.LINEAR)
		drv.matrix_to_texref(img[:,:,1], texrefG, order="F")
		texrefG.set_filter_mode(drv.filter_mode.LINEAR)
		drv.matrix_to_texref(img[:,:,2], texrefB, order="F")
		texrefB.set_filter_mode(drv.filter_mode.LINEAR)
	
		bdim = (16, 16, 1)
		dx, mx = divmod(cols, bdim[0])
		dy, my = divmod(rows, bdim[1])
	
		gdim = ((dx + (mx>0)) * bdim[0], (dy + (my>0)) * bdim[1])
	
		dest = np.zeros((rows,cols,3)).astype("float32")
		m0 = (m[0,:]-1).astype("float32")
		m1 = (m[1,:]-1).astype("float32")
	
		interpolation(drv.Out(dest), drv.In(m0), drv.In(m1), block=bdim, grid=gdim, texrefs=[texrefR, texrefG, texrefB])
	
		return dest.astype("uint8")				
			