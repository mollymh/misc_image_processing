import cv2
import ipcv
import numpy

def histogram(im, maxCount=255, ignoreZero=False):
   numberRows, numberColumns, numberBands, dataType = ipcv.dimensions(im)
   histogram = []
   pdf = []
   cdf = []
   for band in range(numberBands):
      h = numpy.int32(cv2.calcHist([im], [band], None, [maxCount+1],
                                                       [0,maxCount+1]))
      if ignoreZero:
         h[0, 0] = 0
      histogram.append(h)
      pdf.append(histogram[-1] / numpy.sum(histogram[-1]).astype(numpy.float64))
      cdf.append(numpy.cumsum(pdf[-1]))
   return numpy.asarray(histogram), numpy.asarray(pdf), numpy.asarray(cdf)


if __name__ == '__main__':

   import matplotlib.pyplot
   import os.path
   import time

   home = os.path.expanduser('~')
   path = os.path.join(home, 'src', 'python', 'examples', 'data')
   filename = os.path.join(path, 'redhat.ppm')
   filename = os.path.join(path, 'crowd.jpg')
   filename = os.path.join(path, 'checkerboard.tif')

   im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
   print('Filename = {0}'.format(filename))
   print('Data type = {0}'.format(type(im)))
   print('Image shape = {0}'.format(im.shape))
   print('Image size = {0}'.format(im.size))

   startTime = time.time()
   h, pdf, cdf = ipcv.histogram(im)
   print('Elapsed time = {0} [s]'.format(time.time() - startTime))

   matplotlib.pyplot.figure(1, [8, 12])

   matplotlib.pyplot.subplot(3, 1, 1)
   matplotlib.pyplot.ylabel('Number of Pixels')
   matplotlib.pyplot.xlim([0, 255])
   if len(h) == 1:
      matplotlib.pyplot.plot(numpy.arange(256), h[0], 'k')
   else:
      for band in range(len(h)):
         matplotlib.pyplot.plot(numpy.arange(256), h[band])

   matplotlib.pyplot.subplot(3, 1, 2)
   matplotlib.pyplot.ylabel('PDF')
   matplotlib.pyplot.xlim([0, 255])
   if len(pdf) == 1:
      matplotlib.pyplot.plot(numpy.arange(256), pdf[0], 'k')
   else:
      for band in range(len(pdf)):
         matplotlib.pyplot.plot(numpy.arange(256), pdf[band])

   matplotlib.pyplot.subplot(3, 1, 3)
   matplotlib.pyplot.xlabel('Digital Count')
   matplotlib.pyplot.ylabel('CDF')
   matplotlib.pyplot.xlim([0, 255])
   matplotlib.pyplot.ylim([0, 1])
   if len(cdf) == 1:
      matplotlib.pyplot.plot(numpy.arange(256), cdf[0], 'k')
   else:
      for band in range(len(cdf)):
         matplotlib.pyplot.plot(numpy.arange(256), cdf[band])

   matplotlib.pyplot.show()
