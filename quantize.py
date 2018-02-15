import numpy
from types import *

def quantize(im, levels, qtype='uniform', maxCount=255, displayLevels=None):
	"""
	Title: quantize
	Description:
		Returns given image quantized at set number of levels, using either a uniform or IGS method.
	Attributes:
		im - ndarray, can be grayscale or color of any size
		levels - number of levels for image to be quantized to; must be positive integer
		qtype - method of quantization to be implemented. Functionality included for IGS
			and uniform, but if other is specified, defaults to uniform.
		maxCount - maximum code value of pixel; must be positive integer
		displayLevels - levels at which image is displayed; must be positive integer or 'None'.
			If the latter, then defaults to same as maxCount.
	"""

	quantIm = im
	dtype = im.dtype

	if levels <= 0 or type(levels) is not int:
		msg = "Specified levels must be a positive integer."
		raise ValueError(msg)
	if displayLevels <= 0 or type(displayLevels) is not int:
		msg = "Specified display levels must be a positive integer."
		raise ValueError(msg)
	if maxCount <= 0 or type(maxCount) is not int:
		msg = "Specified maximum digital count must be a positive integer."
		raise ValueError(msg)
	if type(im) is not numpy.ndarray:
		msg = "Specified image type must be ndarray."
		raise TypeError(msg)
	if qtype != 'igs' and qtype != 'uniform':
		msg = "Quantization types available are uniform and IGS. Defaulting to Uniform."
		print(msg)

	binsize = (maxCount + 1)/levels
	remainder = 0

	if qtype == 'igs':
		for j in range(im.size):
			px = quantIm.flat[j]
			if (px + remainder) >= maxCount:
				px = maxCount
			else:
				px += remainder
			remainder = px%binsize
			quantIm.flat[j] = px

	quantIm = quantIm//binsize
	quantIm = quantIm*binsize
	quantIm = quantIm.astype(dtype)

	return quantIm

if __name__ == '__main__':

	import cv2
	import ipcv
	import os.path

	home = os.path.expanduser('~')
	filename = home + os.path.sep + 'src/python/examples/data/crowd.jpg'
	filename = home + os.path.sep + 'src/python/examples/data/redhat.ppm'
	filename = home + os.path.sep + 'src/python/examples/data/linear.tif'
#	filename = home + os.path.sep + 'src/python/examples/data/lenna.tif'

	im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
	print('Filename = {0}'.format(filename))
	print('Data type = {0}'.format(type(im)))
	print('Image shape = {0}'.format(im.shape))
	print('Image size = {0}'.format(im.size))

	cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
	cv2.imshow(filename, im)

	numberLevels = 7
	quantizedImage = ipcv.quantize(im,
                                        numberLevels,
                                        qtype='uniform',
                                        displayLevels=256)
	cv2.namedWindow(filename + ' (Uniform Quantization)', cv2.WINDOW_AUTOSIZE)
	cv2.imshow(filename + ' (Uniform Quantization)', quantizedImage)

	numberLevels = 7
	quantizedImage = ipcv.quantize(im,
                                        numberLevels,
                                        qtype='igs',
                                        displayLevels=256)
	cv2.namedWindow(filename + ' (IGS Quantization)', cv2.WINDOW_AUTOSIZE)
	cv2.imshow(filename + ' (IGS Quantization)', quantizedImage)

	action = ipcv.flush()
