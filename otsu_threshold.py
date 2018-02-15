import ipcv
import numpy
import matplotlib.pyplot
import math

def otsu_threshold(im, maxCount=255, verbose=False):
	"""
	Title: otsu_threshold
	Author: Molly Hill, mmh5847@rit.edu
	Description:
		Finds threshold for foreground/background in given image
	Returns:
		matte - binary image indicating areas above/below threshold
		kOpt - optimum threshold for given image
	Attributes:
		im - source image; ndarray, must be grayscale
		maxCount - maximum code value of pixel; must be positive integer
		verbose - if True, plots histogram of image, marked with optimum threshold
	Requires:
		histogram.py, author: Carl Salvaggio
	"""

	if maxCount <= 0 or type(maxCount) is not int:
		msg = "Specified maximum digital count must be a positive integer."
		raise ValueError(msg)
	if type(im) is not numpy.ndarray:
		msg = "Specified image type must be ndarray."
		raise TypeError(msg)
	if type(verbose) is not bool:
		msg = "Verbose must be a boolean. Defaulting to False."
		print(msg)

	#initialize variables
	srcIm = numpy.copy(im)
	histo = ipcv.histogram(srcIm)[0]
	srcPDF = ipcv.histogram(srcIm)[1][0]
	matte = []
	stdDev = []
	kOpt = 0
	firstMtot = []
	
	
	indices = numpy.arange(maxCount+1)
	for i in range(numpy.size(indices)):
		firstMtot.extend([srcPDF[i]*indices[i]]) #pre-sum first moment across image
	meanLvl = numpy.sum(firstMtot) #mean level of image
	
	for k in range(0,maxCount):
		zeroM = numpy.sum(srcPDF[0:k]) #zero moment
		if zeroM == 0 or zeroM == 1:
			stdDev.extend([0])
		else:
			firstM = numpy.sum(firstMtot[:k]) #first moment up to k
			stdDev.extend([(((meanLvl*zeroM) - firstM)**2)//(zeroM*(1-zeroM))])		
	kOpt = numpy.argmax(stdDev) + 1
	
	matte = numpy.uint8(1*(im >= kOpt)) #get as Booleans and convert to integers
	
	if verbose == True: #plot histogram and threshold
		#NOTE THIS SLOWS EVERYTHING DOWN BY A LOT
		matplotlib.pyplot.plot(indices,histo[0])
		matplotlib.pyplot.ylabel('Number of Pixels')
		matplotlib.pyplot.xlabel('Digital Count')
		matplotlib.pyplot.ylim([0,max(histo[0])])
		matplotlib.pyplot.xlim([0,maxCount])
		matplotlib.pyplot.axvline(x=kOpt, label='Threshold'+str(kOpt),color='r')
		matplotlib.pyplot.show()

	return matte, kOpt


if __name__ == '__main__':

	import cv2
	import ipcv
	import os.path
	import time

	home = os.path.expanduser('~')
	filename = home + os.path.sep + 'src/python/examples/data/giza.jpg'

	im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
	print('Filename = {0}'.format(filename))
	print('Data type = {0}'.format(type(im)))
	print('Image shape = {0}'.format(im.shape))
	print('Image size = {0}'.format(im.size))

	startTime = time.time()
	thresholdedImage, threshold = ipcv.otsu_threshold(im, verbose=True)
	print('Elapsed time = {0} [s]'.format(time.time() - startTime))

	print('Threshold = {0}'.format(threshold))

	cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
	cv2.imshow(filename, im)
	cv2.namedWindow(filename + ' (Thresholded)', cv2.WINDOW_AUTOSIZE)
	cv2.imshow(filename + ' (Thresholded)', thresholdedImage * 255)

	action = ipcv.flush()
