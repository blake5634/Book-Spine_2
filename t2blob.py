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
    #gimg = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, bimg = cv2.threshold(im,127,255,cv2.THRESH_BINARY)  
    #contours, hierarchy = cv2.findContours(bimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(bimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print('Found {:} contours'.format(len(contours)))
    im2 = im.copy()
    
    origcontours = []
    bookcontours = []
    boxycontours = []
    othercontours = []
    

    ndc = 0
    nlc = 0
    for i in range(len(contours)):
        #print('\nLooking at Contour ',i)
        nlc +=1
        c = contours[i]
        moms = cv2.moments(c)
        perim = cv2.arcLength(c, True)
        area  = cv2.contourArea(c)
        #elong = perim / np.sqrt(area)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        elong = box_elong(box)
        boxiness = boxy(c,box)
        #hull  = cv2.convexHull(c)
        #if elong > 7 and elong < 12 and  area > 5000:
        if area > bpar.area_min  and elong > bpar.elong_min  and elong < bpar.elong_max:
            if area > 2.0*(bpar.area_min) :
                col = bpar.colors['red']
            else:
                col = bpar.colors['green']
              
            origcontours.append(c)
            #cv2.drawContours(blank, [box], -1, col, 3)
            bookcontours.append([box])
            ndc += 1
        elif area > bpar.area_min/2 and boxiness >= bpar.enough_corners:
            boxycontours.append([c])
            bookcontours.append([box])
            ndc += 1
            
        else:
            if area > bpar.noise_area_threshold:
                othercontours.append(c)

        #
        #   find "boxy" blobs that are smaller than area_min
        #
        
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
    print('\n\n   Looked at {:} unfiltered contours.\n\n'.format(nlc))
    return bookcontours, boxycontours, origcontours, othercontours
    
    
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

##############
#
#  does the contour resemble part of a rectangle??
#   (e.g. a couple of corners close to bounding rectangle)
#
def boxy(blob, box):
    dmins = np.zeros(4)
    for i,d in enumerate( dmins ):
        dmins[i] = 999999999
        
    for i, corner in enumerate(box):
        for pl in blob:
            p = pl[0] 
            di = pdist(p,corner)
            if di < dmins[i]:
                dmins[i] = di
    th = bpar.corner_dist_max_px
    nclose = 0
    for i in range(4):
        if dmins[i] < th:
            nclose += 1
    return nclose

            

def boxy0(blob, box):
    rmin_box = 999999999
    rmax_box = 0
    cmin_box = rmin_box
    cmax_box = 0
    rmin_blob = 999999999
    rmax_blob = 0
    cmin_blob = rmin_blob
    cmax_blob = 0
    
    for p in box:
        if p[0] < rmin_box:
            rmin_box = p[0]
        if p[0] > rmax_box:
            rmax_box = p[0]
        if p[1] < cmin_box:
            cmin_box = p[1]
        if p[1] > cmax_box:
            cmax_box = p[1]
        
    for p in blob:
        if p[0] < rmin_blob:
            rmin_blob = p[0]
        if p[0] > rmax_blob:
            rmax_blob = p[0]
        if p[1] < cmin_blob:
            cmin_blob = p[1]
        if p[1] > cmax_blob:
            cmax_blob = p[1]
        
    box_corners = [[rmin_box, cmin_box],[rmin_box, cmax_box],[rmax_box, cmin_box],[rmax_box, cmax_box]]
    blob_corners = [[rmin_blob, cmin_blob],[rmin_blob, cmax_blob],[rmax_blob, cmin_blob],[rmax_blob, cmax_blob]]
    
    score = 0
    for xc in box_corners:
        for lc in blob_corners:
            if pdist(xc,lc) < 10:
                score += 1
    return score

def box_elong(b):
    l1 = pdist(b[0],b[1])
    l2 = pdist(b[0],b[3])
    if l2 >= 1.0 and l1 > 0:
        e = l1/l2
    else:
        return 999999999.999
    if e< 1:
        e = 1.0/e
    return e

def pdist(p1,p2):
    d1 = p1[0]-p2[0]
    d2 = p1[1]-p2[1]
    r = np.sqrt(d1*d1+d2*d2)
    return r

