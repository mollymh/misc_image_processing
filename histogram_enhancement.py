import ipcv
import numpy

def histogram_enhancement(im, etype='linear2', target=None, maxCount=255):
	"""
	Title: histogram_enhancement
	Author: Molly Hill, mmh5847@rit.edu
	Description:
		Returns given image quantized at set number of levels, using either a uniform or IGS method.
	Attributes:
		im - ndarray, can be grayscale or color of any size
		etype - enhancement type, of the following options:
			    - 'linearX' where X is an integer percent of the area to be
			       clipped/crushed with contrast increas
			    - 'match' where the image is to be matched to a provided
			       target image or PDF
			    - 'equalize' where the image's histogram is to be spread
			       equally across digital count
		target - if etype = 'match', then target must be provided as either an
			 image (3D array) or PDF (1D array)
		maxCount - maximum code value of pixel; must be positive integer
	Requires:
		histogram.py, author: Carl Salvaggio
		dimensions.py, author: Carl Salvaggio
	"""

	if maxCount <= 0 or type(maxCount) is not int:
		msg = "Specified maximum digital count must be a positive integer."
		raise ValueError(msg)
	if type(im) is not numpy.ndarray:
		msg = "Specified image type must be ndarray."
		raise TypeError(msg)
	if etype[:6] != 'linear' and etype != 'match' and etype != 'equalize':
		msg = "Enhancement types available are linear, match, and equalize. Defaulting to linear2."
		print(msg)
	if etype == 'match':
		if type(target) == None:
			etype = 'equalize'
			msg = "If using match, target must be provided."
			print(msg)
		elif target.ndim !=1 and target.ndim !=2:
			print(target.ndim)
			msg = "Provided target must be PDF (1-D array) or image (3-D array)"
			raise TypeError(msg)

	enhIm = numpy.copy(im)
	srcCDF = ipcv.histogram(enhIm)[2]
	DCout = []
	tgtCDF = []
	tgtPDF = target
	
	if etype == 'match' or etype == 'equalize':
		if etype == 'match' and target.ndim != 1: #is image
			tgtCDF = ipcv.histogram(target)[2][0] #create CDF of target, currently does red channel if color
		else: #equalize or PDF passed in as target for matching
			if etype == 'equalize':
				tgtPDF = numpy.ones(maxCount+1)/(maxCount+1)
			tgtCDF = numpy.cumsum(tgtPDF) #convert PDF to CDF

		for i in range(ipcv.dimensions(srcCDF)[1]): #createLUT
			difference = numpy.fabs(numpy.subtract(tgtCDF,srcCDF[0][i])) #red channel only
			DCout.extend([int(maxCount*tgtCDF[numpy.argmin(difference)])])
		for j in range(im.size): #apply LUT
			enhIm.flat[j] = DCout[enhIm.flat[j]] #uses original code value to assign new output from LUT

	else: #linear
		pct = (int(etype[6:])/2) / 100 #extract percent from etype and halve
		difference = numpy.fabs(numpy.subtract(srcCDF[0],pct))
		DCmin = numpy.argmin(difference)
		difference = numpy.fabs(numpy.subtract(srcCDF[0],(1-pct)))
		DCmax = numpy.argmin(difference)

		slope = (maxCount+1)/(DCmax-DCmin)
		intercept = -slope * DCmin
		
		for j in range(enhIm.size):
			px = enhIm.flat[j]
			if px >= DCmax:
				px = maxCount
			elif px <= DCmin:
				px = 0
			else:
				px = slope * px + intercept
			enhIm.flat[j] = int(px)

	return enhIm

if __name__ == '__main__':

	import cv2
	import os.path
	import time

	home = os.path.expanduser('~')
	path = os.path.join(home, 'src', 'python', 'examples', 'data')
	filename = os.path.join(path, 'lenna.tif')
	filename = os.path.join(path, 'giza.jpg')
	filename = os.path.join(path, 'crowd.jpg')
	filename = os.path.join(path, 'redhat.ppm')

	matchFilename = os.path.join(path, 'redhat.ppm')
	matchFilename = os.path.join(path, 'crowd.jpg')
	matchFilename = os.path.join(path, 'giza.jpg')
	matchFilename = os.path.join(path, 'lenna.tif')

	im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
	print('Filename = {0}'.format(filename))
	print('Data type = {0}'.format(type(im)))
	print('Image shape = {0}'.format(im.shape))
	print('Image size = {0}'.format(im.size))

	cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
	cv2.imshow(filename, im)

	print('Linear 2% ...')
	startTime = time.time()
	enhancedImage = ipcv.histogram_enhancement(im, etype='linear2')
	print('Elapsed time = {0} [s]'.format(time.time() - startTime))
	cv2.namedWindow(filename + ' (Linear 2%)', cv2.WINDOW_AUTOSIZE)
	cv2.imshow(filename + ' (Linear 2%)', enhancedImage)

	print('Linear 1% ...')
	startTime = time.time()
	enhancedImage = ipcv.histogram_enhancement(im, etype='linear1')
	print('Elapsed time = {0} [s]'.format(time.time() - startTime))
	cv2.namedWindow(filename + ' (Linear 1%)', cv2.WINDOW_AUTOSIZE)
	cv2.imshow(filename + ' (Linear 1%)', enhancedImage)

	#print('Equalized ...')
	#startTime = time.time()
	#enhancedImage = ipcv.histogram_enhancement(im, etype='equalize')
	#print('Elapsed time = {0} [s]'.format(time.time() - startTime))
	#cv2.namedWindow(filename + ' (Equalized)', cv2.WINDOW_AUTOSIZE)
	#cv2.imshow(filename + ' (Equalized)', enhancedImage)

	#tgtIm = cv2.imread(matchFilename, cv2.IMREAD_UNCHANGED)
	#print('Matched (Image) ...')
	#startTime = time.time()
	#enhancedImage = ipcv.histogram_enhancement(im, etype='match', target=tgtIm)
	#print('Elapsed time = {0} [s]'.format(time.time() - startTime))
	#cv2.namedWindow(filename + ' (Matched - Image)', cv2.WINDOW_AUTOSIZE)
	#cv2.imshow(filename + ' (Matched - Image)', enhancedImage)

	#tgtPDF = numpy.ones(256) / 256
	#print('Matched (Distribution) ...')
	#startTime = time.time()
	#enhancedImage = ipcv.histogram_enhancement(im, etype='match', target=tgtPDF)
	#print('Elapsed time = {0} [s]'.format(time.time() - startTime))
	#cv2.namedWindow(filename + ' (Matched - Distribution)', cv2.WINDOW_AUTOSIZE)
	#cv2.imshow(filename + ' (Matched - Distribution)', enhancedImage)

	action = ipcv.flush()

