import cv2

def flush():
   print('Press "c" to continue, ESC to exit')
   delay = 100
   while True:
      k = cv2.waitKey(delay)

      # ESC pressed
      if (k & 0xff) == 27:
         action = 'exit'
         print('Exiting ...')
         break

      # c or C pressed
      if (k & 0xff) == 99 or (k & 0xff) == 67:
         action = 'continue'
         print('Continuing ...')
         break

   return action
