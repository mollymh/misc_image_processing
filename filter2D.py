import ipcv
import numpy as np

def filter2D(src, dstDepth, kernel, delta=0, maxCount=255):
	"""
	Title:
		filter2D
	Description:
		Given source image and destination ('map') shape, and a set of
		corners for both, will create a map to align the source image
		with the destination.
	Attributes:
		src - source image to be filtered
		dstDepth - data type of image to be returned
		kernel - filter matrix to be convolved across image
		delta - gray level offset for final image
		maxCount - maximum code value of image; must be positive integer
	Author:
		Molly Hill, mmh5847@rit.edu
	"""

	#error-checking
	if type(src) is not np.ndarray:
		raise TypeError("Source image type must be ndarray.")
	if type(dstDepth) is not type:
		raise TypeError("dstDepth must be a data type.")
	if type(kernel) is not np.ndarray:
		raise TypeError("Kernel type must be ndarray.")	
	if maxCount <= 0 or type(maxCount) is not int:
		raise ValueError("Specified maximum digital count must be a positive integer.")

	x = kernel.shape[0]//2
	y = kernel.shape[1]//2
	dim = [(x,x),(y,y)]
	if len(src.shape) == 3: #so color image has proper padding
		dim.append((0,0))
		
	#pads image to deal with edges
	im = np.pad(src,dim,'edge').astype(np.uint16)
	
	if np.sum(kernel) != 0:
		kernel = kernel/np.sum(kernel) #normalizes filter by weight

	#creates list of tuples of shifts for kernel application
	r = []
	for a in range(y,-y-1,-1):
		for b in range(x,-x-1,-1):
			r.append((b,a))

	#shifts and multiplies kernel, while summing for result
	resBd = np.zeros(im.shape)
	for i in range(kernel.size):
		resBd += np.roll(im,r[i],(1,0)) * kernel.flat[i]

	#removes padding, adds offset, and sets data type
	dst = (np.clip(resBd[x:-x,y:-y]+delta,0,maxCount)).astype(dstDepth)
	
	return dst
	

if __name__ == '__main__':

	import cv2
	import os.path
	import time

	home = os.path.expanduser('~')
	filename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'
	filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
	filename = home + os.path.sep + 'src/python/examples/data/checkerboard.tif'
	filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'

	src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

	dstDepth = ipcv.IPCV_8U
	#kernel = np.asarray([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
	#offset = 0
	#kernel = np.asarray([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
	#offset = 128
	#kernel = np.ones((15,15))
	#offset = 0
	kernel = np.asarray([[1,1,1],[1,1,1],[1,1,1]])
	offset = 0

	startTime = time.time()
	dst = ipcv.filter2D(src, dstDepth, kernel, delta=offset)
	print('Elapsed time = {0} [s]'.format(time.time() - startTime))

	cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
	cv2.imshow(filename, src)

	cv2.namedWindow(filename + ' (Filtered)', cv2.WINDOW_AUTOSIZE)
	cv2.imshow(filename + ' (Filtered)', dst)

	action = ipcv.flush()
