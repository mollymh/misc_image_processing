def dimensions(im):
   dimensions = im.shape
   numberRows = dimensions[0]
   numberColumns = dimensions[1]
   if len(dimensions) == 3:
      numberBands = dimensions[2]
   else:
      numberBands = 1
   dataType = im.dtype
   return numberRows, numberColumns, numberBands, dataType


if __name__ == '__main__':

   import cv2
   import ipcv
   import os.path

   home = os.path.expanduser('~')
   path = os.path.join(home, 'src', 'python', 'examples', 'data')
   filename = os.path.join(path, 'lenna.tif')
   filename = os.path.join(path, 'redhat.ppm')
   filename = os.path.join(path, 'crowd.jpg')
   filename = os.path.join(path, 'checkerboard.tif')

   im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
   print('Filename = {0}'.format(filename))
   print('Data type = {0}'.format(type(im)))
   print('Image shape = {0}'.format(im.shape))
   print('Image size = {0}'.format(im.size))

   print(ipcv.dimensions(im))
