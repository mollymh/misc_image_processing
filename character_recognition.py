import numpy as np
import cv2
import math as m

def character_recognition(src,
				templates,
				codes,
				threshold=-5,
				filterType='spatial'):
	"""
	Title:
		character_recognition
	Description:
		Identifies and counts designated characters in given source image.
	Attributes:
		src - source image in which characters are to be identified
		templates - templates for each character
		codes - ASCII codes corresponding to each character in templates
		threshold - threshold for valid identification
		filterType - method of recgonition - either 'spatial' or 'matched':
				-'spatial': Uses 2D spatial filter
				-'matched': Uses vectorized angle-minimization
	Author:
		Molly Hill, mmh5847@rit.edu
	"""

	#error-checking
	if type(src) != np.ndarray:
		raise TypeError("Source image type must be ndarray.")
	if len(templates) != len(codes):
		raise ValueError("ASCII codes must be provided for all provided character templates, vice versa.")
	if threshold < 0: #threshold not provided
		print("Proper threshold not provided. Defaulting based on filterType." + "\n")
		if filterType == 'spatial':
			threshold = 0.000001
		if filterType == 'matched':
			threshold = 0.9919
	if (type(filterType) is not str) or (filterType != 'spatial' and filterType != 'matched'):
		raise ValueError("filterType must be string specifying either 'spatial' or 'matched' method")


	#initialize list variables
	letters = []
	results = []

	#get max DC for normalization
	bitDepth = int(str(src.dtype)[4:])
	maxCount = int(m.pow(2,bitDepth)) - 1

	#normalize image and fix datatypes
	src = (src/maxCount).astype(np.float64)
	templates = templates.astype(np.float64)
	
	#pre-calculate row and column info for source and templates
	rdim = templates[0].shape[0]
	cdim = templates[0].shape[1]
	nrow = src.shape[0]/rdim
	ncol = src.shape[1]/cdim
	

	for x in range(len(templates)): #iterate through characters

		if filterType == 'spatial':
			char = (maxCount-templates[x])/maxCount #invert/norm char template
			resp = cv2.filter2D(src, -1, char)/maxCount #filter src w/template
			matches = np.where(resp <= threshold)
			count = len(matches[0])
			for i in range(count): #store linear coordinate and letters for later
				letters += [(int(matches[0][i]*ncol + matches[1][i]),chr(codes[x]))]

			
		if filterType == 'matched':
			count = 0
			for r in range(src.shape[0]): #iterate through image
				for c in range(src.shape[1]):
					s = src[r:r+rdim,c:c+cdim].flatten() #slice template-sized section
					char = templates[x].flatten()
					if len(s) != len(char): #at edge of image
						break
					#vectorized comparison slice to template
					resp = np.dot(s,char)/(m.sqrt(np.dot(s,s))*m.sqrt(np.dot(char,char)))
					if resp >= threshold:
						count+=1
						#store linear coordinate and letter for later
						letters += [(int(r*ncol + c),chr(codes[x]))]

		#store number of letter occurrences
		results += [(chr(codes[x]),count)]


	#creates structured array to sort text	
	dtype = [('location',int),('letter','S10')]
	lets= np.array(letters,dtype=dtype)
	lets = np.sort(lets,order='location')
	locs = lets[['location']].astype(int) #fixing data types
	lets = lets[['letter']].astype(str)
	text = ""
	for y in range(len(locs)): #forms text string
		if locs[y]-locs[y-1] > cdim*1.5: #deals with spaces
			text += " "
		text += lets[y]

	#creates structured array in case codes are out-of-order
	dtype = [('letter','S10'),('count',int)]
	histogram = np.array(results,dtype=dtype)
	histogram = np.sort(histogram,order='letter') #sorts by ASCII code
	histogram = [list(histogram[['letter']].astype(str)),list(histogram[['count']].astype(int))]
	
	
	return text, histogram
			

if __name__ == '__main__':

	import cv2
	import fnmatch
	import numpy
	import os
	import os.path
	import matplotlib.pyplot as plt

	home = os.path.expanduser('~')
	baseDirectory = home + os.path.sep + 'src/python/examples/data'
	baseDirectory += os.path.sep + 'character_recognition'

	documentFilename = baseDirectory + '/notAntiAliased/text.tif'
	#documentFilename = baseDirectory + '/notAntiAliased/alphabet.tif'
	charactersDirectory = baseDirectory + '/notAntiAliased/characters'

	document = cv2.imread(documentFilename, cv2.IMREAD_UNCHANGED)

	characterImages = []
	characterCodes = []
	for root, dirnames, filenames in os.walk(charactersDirectory):
		for filename in sorted(filenames):
			currentCharacter = cv2.imread(root + os.path.sep + filename,
						cv2.IMREAD_UNCHANGED)
			characterImages.append(currentCharacter)
			code = int(os.path.splitext(os.path.basename(filename))[0])
			characterCodes.append(code)
	characterImages = numpy.asarray(characterImages)
	characterCodes = numpy.asarray(characterCodes)

	# Define the filter threshold
	threshold = 0.000001

	text, histogram = character_recognition(document, 
						characterImages, 
						characterCodes, 
						threshold, 
						filterType='spatial')

	print("Text using spatial method: " + text + "\n")
	print("Close plot when finished..." + "\n" + "\n")
	
	x = np.arange(len(histogram[0]))
	plt.bar(x,histogram[1],align='center')
	plt.xticks(x,histogram[0])
	plt.xlabel('Letter')
	plt.ylabel('Occurrence')
	plt.title('Letter Occurence in ' + (str(documentFilename)[len(baseDirectory):]) + " (spatial method)")
	plt.show()
	
	# Define the filter threshold
	threshold = 0.9919

	text, histogram = character_recognition(document, 
						characterImages, 
						characterCodes, 
						threshold, 
						filterType='matched')

	print("Text using match method: " + text + "\n")
	print("Close plot when finished..." + "\n" + "\n")
	
	x = np.arange(len(histogram[0]))
	plt.bar(x,histogram[1],align='center')
	plt.xticks(x,histogram[0])
	plt.xlabel('Letter')
	plt.ylabel('Occurrence')
	plt.title('Letter Occurence in ' + (str(documentFilename)[len(baseDirectory):]) + " (match method)")
	plt.show()
