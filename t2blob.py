import cv2
import numpy as np
from PIL import Image
import sys

import book_parms as bpar
import book_classes as bc

def getBookBlobs(imC):
    im = imC.image
    #blank = imC.blank()
    blank = cv2.imread('tcolor.png')
    gimg = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, bimg = cv2.threshold(gimg,127,255,cv2.THRESH_BINARY)  
    contours, hierarchy = cv2.findContours(bimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print('Found {:} contours'.format(len(contours)))
    im2 = im.copy()
    
    origcontours = []
    bookcontours = []
    othercontours = []
    

    ndc = 0
    for i in range(len(contours)):
        print('\nLooking at Contour ',i)
        c = contours[i]
        moms = cv2.moments(c)
        perim = cv2.arcLength(c, True)
        area  = cv2.contourArea(c)
        elong = perim / np.sqrt(area)
        #hull  = cv2.convexHull(c)
        #if elong > 7 and elong < 12 and  area > 5000:
        if area > 3000 and elong > 5 and elong < 15:
            if   area > 4000:
                col = bpar.colors['red']
            else:
                col = bpar.colors['green']
              
            origcontours.append(c)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #cv2.drawContours(blank, [box], -1, col, 3)
            bookcontours.append([box])
            ndc += 1
        else:
            if area > 100:
                othercontours.append(c)

        ##print('Moments:     ', moms)
        #print('Area:        {:8.1f}'.format(  area))
        #print('Perimeter    {:8.1f}'.format( perim))
        #print('Elongation   {:8.3f}'.format( elong))
        
    #if ndc > 0:
        #title = 'contours, hulls'
        #cv2.imshow(title,blank)
        #cv2.waitKey(-1)
    #else:
        #print('\n\n              No Contours Match \n\n')

    return bookcontours, origcontours, othercontours
    
    
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
    #if len(sys.argv) != 2:
        #print ('supply an arg')
        #quit()
        
    bookcs = []
    for i in range(20):
        fname = 'blobSet{:02d}.png'.format(int(i))
        print('Opening: ', fname)
        im1 = cv2.imread(fname) 
        im1Image = bc.bookImage(im1,bpar.scale)
        
        bcont =  getblobs(im1Image)
        for b in bcont:
            bookcs.append(b)
            
    print ('\n\n  Found {:} book contours \n\n'.format(len(bookcs)))
    
    col = bpar.colors['red']
    blank = cv2.imread('tcolor.png')
    for b in bookcs:
        cv2.drawContours(blank, b, -1, col, 3)

    cv2.imshow('title',blank)
    cv2.waitKey(-1)

