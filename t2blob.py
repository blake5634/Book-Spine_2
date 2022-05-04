import cv2
import numpy as np
from PIL import Image
import sys

import book_parms as bp


def getblobs(im):
    ## convert to BW image
    #if Image.isImageType(im):
        #im = np.array(im.convert("L"))
    #if isinstance(im, np.ndarray):
        #if (len(im.shape) >= 3 
        #and im.shape[2] > 1):
            #im = im[:,:,0]
    gimg = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, bimg = cv2.threshold(gimg,127,255,cv2.THRESH_BINARY)    
    contours, hierarchy = cv2.findContours(bimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print('Found {:} contours'.format(len(contours)))
    im2 = im.copy()

    for i in range(len(contours)):
        print('\nLooking at Contour ',i)
        c = contours[i]
        moms = cv2.moments(c)
        perim = cv2.arcLength(c, True)
        area  = cv2.contourArea(c)
        elong = perim / area
        hull  = cv2.convexHull(c)
        if elong < 0.2:
            col = bp.colors['red']
        else:
            col = bp.colors['green']
            
        ##cv2.drawContours(im2,hull, -1, col, 3)
        #if i == 3 or i ==4:
            #col = bp.colors['green']
        cv2.drawContours(im2, c, -1, col, 3)

        #print('Moments:     ', moms)
        print('Area:        {:8.1f}'.format(  area))
        print('Perimeter    {:8.1f}'.format( perim))
        print('Elongation   {:8.3f}'.format( elong))
        
    title = 'contours, hulls'
    cv2.imshow(title,im2)
    cv2.waitKey(-1)

    
    
###
##
#  2D feature calls
#   findContours(binaryImage, contours, RETR_LIST, CHAIN_APPROX_NONE);
#   moments()    Moments moms = moments(contours[contourIdx]);
#   perimeter = arcLength(contours[contourIdx], true);
#   convexHull(contours[contourIdx], hull);
#    double area = contourArea(contours[contourIdx]);
    #double hullArea = contourArea(hull);
#

####

if __name__== '__main__':
    if len(sys.argv) != 2:
        print ('supply an arg')
        quit()
    im1 = cv2.imread(sys.argv[1]) 
    
    blobs = getblobs(im1)
