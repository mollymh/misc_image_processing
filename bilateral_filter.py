import numpy as np
import math as m
import ipcv
import cv2

def bilateral_filter(src,
			sigmaDistance,
			sigmaRange,
			d=-1,
			borderType=ipcv.BORDER_WRAP,
			maxCount=255):
	"""
	Title:
		bilateral_filter
	Description:
		Implements bilateral filter, reducing noise while preserving edges.
	Attributes:
		src - source image to be filtered
		sigmaDistance - one std. dev. for closeness values
		sigmaRange - one std. dev. for color difference values
		d - number of deviations
		borderType - border treatment for padding
		maxCount - maximum code value for output image
	Author:
		Molly Hill, mmh5847@rit.edu
	"""

	#error-checking
	if type(src) is not np.ndarray:
		raise TypeError("Source image type must be ndarray.")
	if type(sigmaDistance) is not int:
		raise TypeError("sigmaDistance must be integer.")
	if type(sigmaRange) is not int:
		raise TypeError("sigmaRange must be integer.")
	if type(d) is not int:
		raise TypeError("sigmaDistance must be integer.")
	if type(borderType) is not int or (borderType < 0 or borderType > 4):
		raise TypeError("borderType must be positive int from 0-4.")
	if maxCount <= 0 or type(maxCount) is not int:
		raise ValueError("Specified maximum digital count must be a positive integer.")
	
	
	#INITIALIZE ARRAYS
	if d <= 0:
		radius = 2*sigmaDistance
	else:
		radius = d
	filterD = np.zeros((2*radius+1,2*radius+1)) #allow for center pixel
	filterR = np.copy(filterD).astype(np.float32)
	dst = np.zeros(src.shape)

	#CLOSENESS FILTER
	diag = m.sqrt(2)
	cent = (radius-1)/2
	#first set distances for each filter location
	for x in range(filterD.shape[0]): #complicated so minimizes sqrt calcs
		for y in range(filterD.shape[1]):
			if x == y:
				if x != cent:
					dV = diag*(m.fabs(y-cent))	
			elif x == cent:
				dV = m.fabs(y-cent)
			elif y == cent:
				dV = m.fabs(x-cent)
			else:
				dV = m.sqrt(m.fabs(x-cent)**2+m.fabs(y-cent)**2)
			#now convert distances to gaussian weights
			filterD[x][y] = dV
			#filterD[x][y] = m.exp(-0.5*(dV/sigmaDistance)**2)
	print(filterD.astype(int))

	#PAD IMAGE
	x = filterD.shape[0]//2
	y = filterD.shape[1]//2
	dim = [(x,x),(y,y)]
	if len(src.shape) == 3: #so color image has proper padding
		dim.append((0,0))
	if borderType == 0:
		bt = 'constant'
	elif borderType == 1: #replicate
		bt = 'edge'
	elif borderType == 2: #reflect
		bt = 'symmetric'
	elif borderType == 3: #wrap
		bt = 'wrap'
	else: # borderType == 4, reflect101
		bt = 'reflect'

	if len(src.shape) == 3: #space conversion for color image
		padIm = np.pad(src,dim,bt).astype(np.uint8)
		padIm = cv2.cvtColor(padIm,cv2.COLOR_BGR2LAB)
	else:
		padIm = np.pad(src,dim,bt).astype(np.float32)


	#IMPLEMENT BILATERAL FILTER
	for r in range(x,src.shape[0]+x): #accounts for padding
		for c in range(y, src.shape[1]+y):
			squiggle = padIm[r-x:r+x+1,c-y:c+y+1] #slice neighborhood
			px = padIm[r][c] #reference pixel
			
			#Create range filter by calculating color differences
			if len(src.shape) == 2:
				diffR = squiggle-px
			else:
				diffR = np.sqrt((squiggle[:,:,0]-px[0])**2+(squiggle[:,:,1]-px[1])**2+(squiggle[:,:,2]-px[2])**2)
			filterR = np.exp(-0.5*(diffR/sigmaRange)**2)

			biFilt = filterD*filterR #multiply filters together
			#integrate and normalize
			if len(src.shape) == 3: #color
				squiggle = cv2.cvtColor(squiggle,cv2.COLOR_LAB2BGR)
				dstC = []
				for chan in range(src.shape[2]):
					dstC.append(np.sum(np.sum(squiggle[:,:,chan]*biFilt,0),0)/np.sum(biFilt))
			else: #grayscale
				dstC = np.sum(squiggle*biFilt)/np.sum(biFilt)
			dst[r-x][c-y] = dstC

	dst = np.clip(dst,0,maxCount).astype(np.uint8) #clip and fix datatype
	
	return dst

if __name__ == '__main__':

	import cv2
	import ipcv
	import os.path
	import time


	home = os.path.expanduser('~')
	#filename = home + os.path.sep + 'src/python/examples/data/panda_color.jpg'
	#filename = home + os.path.sep + 'src/python/examples/data/cartoon_orig.jpg'
	#filename = home + os.path.sep + 'src/python/modules/ipcv/bilateral_images/testIm.png'
	filename = home + os.path.sep + 'src/python/examples/data/panda.jpg'

	src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

	startTime = time.time()
	dst = ipcv.bilateral_filter(src, 3, 10, d=-1)
	print('Elapsed time = {0} [s]'.format(time.time() - startTime))

	cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
	cv2.imshow(filename, src)

	cv2.namedWindow(filename + ' (Bilateral)', cv2.WINDOW_AUTOSIZE)
	cv2.imshow(filename + ' (Bilateral)', dst)

	action = ipcv.flush()

	"""	
	#MY OWN TEST DATA SET
	import numpy as np
	import random
	import matplotlib.pyplot as plt
	from matplotlib import cm
	from mpl_toolkits.mplot3d import Axes3D

	testIm = np.ones((200,200))
	testIm[:,0:100] = testIm[:,0:100]*64
	testIm[:,100:] = testIm[:,100:]*192
	for i in range(testIm.size):
		testIm.flat[i] += random.randint(-16,16)
	testIm = np.clip(testIm,0,255).astype(np.uint8)
	#then I saved the image out
	
	#I created gaussian plots by modifying my function code
	
	home = os.path.expanduser('~')
	filename = home + os.path.sep + 'src/python/modules/ipcv/bilateral_images/testIm.png'
	src = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
	dst = ipcv.bilateral_filter(src, 10, 10, d=-1)
	cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
	
	cv2.imshow(filename, src)
	cv2.namedWindow(filename + ' (Bilateral)', cv2.WINDOW_AUTOSIZE)
	cv2.imshow(filename + ' (Bilateral)', dst)
	
	xy = np.indices((50,50))
	z = dst[75:125,75:125,0]
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(xy[0],xy[1],z,cmap = cm.coolwarm)
	ax.set_zlabel("Code Value")
	ax.set_title("sigmaD = 10, sigmaR = 10")
	plt.show()

	action = ipcv.flush()
	"""
