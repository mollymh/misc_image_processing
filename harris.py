import cv2
import ipcv
import numpy as np

def harris(src, sigma=1, k=0.04):
	"""
	Title:
		harris
	Description:
		Provides map of intensities based on probability of
		a corner in provided image
	Attributes:
		src - source image in which corners are to be detected
		sigma - factor in gaussian used for noise reduction
		k - constant used in responsivity calculation

	Author:
		Molly Hill, mmh5847@rit.edu
	"""

	#error-checking
	if type(src) != np.ndarray:
		raise TypeError("Source image type must be ndarray.")
	if sigma <= 0:
		raise ValueError("Sigma must be int or float > 0.")
	if k <= 0:
		raise ValueError("k must be int or float > 0.")

	#fix image type for most precise calculations
	src = src.astype(np.float64)

	#partial derivatives of image
	grad = np.reshape(np.repeat(np.array([-1,0,1]),3),(3,3))
	Ix = cv2.filter2D(src,-1,grad.T)
	Iy = cv2.filter2D(src,-1,grad)
	
	#reduce noise in image
	w = np.exp(-0.5*((grad.T**2+grad**2)/sigma**2))
	
	#calculate shifts in image
	A = cv2.filter2D(Ix**2,-1,w)
	B = cv2.filter2D(Iy**2,-1,w)
	C = cv2.filter2D(Ix*Iy,-1,w)
	
	#eigenvalues for responses
	Tr = A + B
	Det = (A*B) - (C**2)
	dst = Det - k*(Tr**2)

	return dst

if __name__ == '__main__':

	import os.path
	import time
	import numpy

	home = os.path.expanduser('~')
	filename = home + os.path.sep + 'src/python/examples/data/checkerboard.tif'
	filename = home + os.path.sep + \
			'src/python/examples/data/sparse_checkerboard.tif'

	src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

	sigma = 1
	k = 0.04
	startTime = time.time()
	dst = ipcv.harris(src, sigma, k)
	print('Elapsed time = {0} [s]'.format(time.time() - startTime))

	cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
	cv2.imshow(filename, src)

	if len(src.shape) == 2:
		annotatedImage = cv2.merge((src, src, src))
	else:
		annotatedImage = src
	fractionMaxResponse = 0.25
	annotatedImage[dst > fractionMaxResponse*dst.max()] = [0,0,255]

	cv2.namedWindow(filename + ' (Harris Corners)', cv2.WINDOW_AUTOSIZE)
	cv2.imshow(filename + ' (Harris Corners)', annotatedImage)

	print('Corner coordinates ...')
	indices = numpy.where(dst > fractionMaxResponse*dst.max())
	numberCorners = len(indices[0])
	if numberCorners > 0:
		for corner in range(numberCorners):
			print('({0},{1})'.format(indices[0][corner], indices[1][corner]))

	action = ipcv.flush()
