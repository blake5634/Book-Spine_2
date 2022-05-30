import cv2
import numpy as np
from PIL import Image
import sys
import glob as gb

import book_parms as bpar
import book_classes as bc

class bblob:  # book blobs
    def __init__(self,c):
        self.contour = c
        self.label = []
        self.M = cv2.moments(self.contour)
        if self.M['m00'] != 0:
            cx = int(self.M['m10']/self.M['m00'])
            cy = int(self.M['m01']/self.M['m00'])
        else:
            cx = 20
            cy = 20 
        self.centerpoint = (cx,cy)
        #print('self.contour shape:',self.contour.shape)
        self.perim = cv2.arcLength(self.contour, True)
        self.area  = cv2.contourArea(self.contour)
        self.rect = cv2.minAreaRect(self.contour) 
        self.box = np.int0(cv2.boxPoints(self.rect)) 
        self.elong = box_elong(self.box)
        self.boxiness = boxy(self.contour, self.box)
        
    def __repr__(self):
        s = 'book blob:\n'
        s += '   contour shape:     '+str(self.contour.shape)+'\n'
        s += '   contour length:    {:}\n'.format(len(self.contour))
        s += '   centerpoint:       {:4},{:4}\n'.format(self.centerpoint[0],self.centerpoint[1])
        s += '   area:           {:8}\n'.format(self.area)
        s += '   box:            \n' + str(self.box)+'\n'
        s += '   boxiness:          {:}\n'.format(self.boxiness)
        return s
        
def getBookBlobs(imC):
    if len(imC.ishape()) != 2:
        print('getBookBlobs: input should be gray not BGR')
        quit()
    im = imC.image
    #blank = imC.blank()
    blank = cv2.imread('tcolor.png')
    #gimg = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, bimg = cv2.threshold(im,127,255,cv2.THRESH_BINARY)  
    #contours, hierarchy = cv2.findContours(bimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(bimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    ntotcs = len(contours)
    print('Found {:} contours'.format(ntotcs))
    im2 = im.copy()
    
    bookcontours = []   # maybe book spines
    boxycontours = []   # smaller blobs that are corner like
    othercontours = []  # everything else above noise_area_threshold
    
    ndc = 0
    nlc = 0
    for i in range(len(contours)):
        #print('\nLooking at Contour ',i)
        nlc +=1        
        #print('raw contour shape: ', contours[i].shape)
        c = bblob(contours[i])   # convert to book blob class
        
        #
        #    the most book-like contours
        #
        if c.area > bpar.area_min  and c.elong > bpar.elong_min  and c.elong < bpar.elong_max:
            bookcontours.append(c)
            ndc += 1
        #
        #   smaller but "boxy" contours (parts of spines)
        #
        elif c.area > bpar.boxy_area_min and c.boxiness >= bpar.boxy_threshold:
            boxycontours.append(c) 
            ndc += 1
            
        #
        #   everything else except tiny ones
        #
        elif c.area > bpar.noise_area_threshold:
            othercontours.append(c)
 
    return  ntotcs, bookcontours, boxycontours, othercontours
    
    
##############
#
#  does the contour resemble part of a rectangle??
#   (e.g. a couple of corners close to bounding rectangle)
#

def boxy(blob, box):
    a1 = bpar.boxy_coef_corners
    a2 = bpar.boxy_coef_perim
    a3 = bpar.boxy_coef_area 
    #score = al*boxyPerim(blob, box) + (1.0-al)*boxyCorners(blob,box)
    score = a1*boxyCorners(blob,box) + a2*boxyPerim(blob, box) + a3*boxyArea(blob, box)
    #print('boxiness: {:4.1f} : {:4.1f},{:4.1f},{:4.1f} '.format(score, boxyCorners(blob,box), boxyPerim(blob, box), boxyArea(blob, box)))

    return score
    
def boxyArea(blob, box):
    ascale = 2000      # guestimate??    
    d1 = pdist(box[0],box[1])
    d2 = pdist(box[1],box[2])
    box_area = d1*d2
    adiff = np.minimum(1.0, np.abs((box_area - cv2.contourArea(blob))/ascale) )  # note contuour area < box area for all cases
    return np.minimum(4.0, 4.0*(1.0 - adiff))
    
def boxyPerim(blob, box):
    # use the difference in perimeter between min vol box and
    #   the contour perimeter
    #b1 = boxyOLD(blob,box)
    d1 = pdist(box[0],box[1])
    d2 = pdist(box[1],box[2])
    boxperimeter = float(2*(d1+d2))
    if boxperimeter < 0.001:
        perdiff = 1.0
    else:
        perdiff = np.abs(boxperimeter - cv2.arcLength(blob,True))/boxperimeter
    if perdiff > 1.0:
        perdiff = 1.00
    score = (1-perdiff)*4   # 4 is perfect to match old boxy
    return score

def boxyCorners(blob, box):
    # use the number of box corners which are close to the contour. 
    dmins = np.zeros(4)
    for i,d in enumerate( dmins ):
        dmins[i] = 999999999
        
    for i, corner in enumerate(box):
        for pl in blob[0]:
            #print('boxy: ',corner.shape, pl.shape)
            di = pdist(pl,corner)
            if di < dmins[i]:
                dmins[i] = di
    th = bpar.corner_dist_max_px
    nclose = 0
    for i in range(4):
        if dmins[i] < th:
            nclose += 1
    return nclose

            

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
    
    imagefilename = 'blobSet*_{:}.png'.format(sys.argv[1])
    print('t2blob: looking at : ',imagefilename, ' in ', bpar.tmp_img_path)
        
    filenames = gb.glob(bpar.tmp_img_path+imagefilename)
    if len(filenames) < 1:
        print('There are no files')
        quit() 
        
    bookcs = []
    for fname in filenames:
        #fname = 'blobSet{:02d}.png'.format(int(i))
        print('Opening: ', fname)
        im1 = cv2.imread(fname) 
        im1Image = bc.bookImage(im1,bpar.scale)
        im1Image.image = im1Image.gray()
        n, bookcontours, boxycontours, othercontours =  getBookBlobs(im1Image) 
        bookcs = bookcs + bookcontours
        
        
    print ('\n\n  Found {:} book contours \n\n'.format(len(bookcs)))
    
    c_red = bpar.colors['red']
    c_blu = bpar.colors['blue']
    blank = cv2.imread('tcolor.png')
    for b in bookcs:
        cv2.drawContours(blank, b.contour, -1, c_red, 3)
        cv2.drawContours(blank, [ b.box ], -1, c_blu, 3)

    cv2.imshow('title',blank)
    cv2.waitKey(-1)
