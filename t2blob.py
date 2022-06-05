import cv2
import numpy as np
from PIL import Image
import sys
import glob as gb

import book_parms as bpar
import book_classes as bc

class bblob:  # book blobs
    def __init__(self,c,image):
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
        self.boxiness = boxy(self.contour, self.box, image)
        
    def __repr__(self):
        s = 'book blob:\n'
        s += '   contour shape:     '+str(self.contour.shape)+'\n'
        s += '   contour length:    {:}\n'.format(len(self.contour))
        s += '   centerpoint:       {:4},{:4}\n'.format(self.centerpoint[0],self.centerpoint[1])
        s += '   area:           {:8}\n'.format(self.area)
        s += '   box:            \n' + str(self.box)+'\n'
        s += '   boxiness:          {:}\n'.format(self.boxiness)
        return s
        
def getBookBlobs(imC,imDisplay):
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
        c = bblob(contours[i],imDisplay)   # convert to book blob class
        
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

def boxy(blob, box, image):
    a1 = bpar.boxy_coef_corners
    a2 = bpar.boxy_coef_perim
    a3 = bpar.boxy_coef_area 
    if boxArea(box) > 100:
        #score = al*boxyPerim(blob, box) + (1.0-al)*boxyCorners(blob,box)
        #score = a1*boxyCorners(blob,box) + a2*boxyPerim(blob, box) + a3*boxyArea(blob, box)
        score = a1*boxyCornersSides(blob,box,image) + a2*boxyPerim(blob, box) + a3*boxyArea(blob, box)
        #print('boxiness: {:4.1f} : {:4.1f},{:4.1f},{:4.1f} '.format(score, boxyCorners(blob,box), boxyPerim(blob, box), boxyArea(blob, box)))
        print('boxy score: {:3.1f}'.format(score))
        return score
    else:
        return 0
    
#  areas the same between min area box and contour?
def boxyArea(blob, box):
    ascale = 2000      # guestimate??
    box_area = boxArea(box)
    adiff = np.minimum(1.0, np.abs((box_area - cv2.contourArea(blob))/ascale) )  # note contuour area < box area for all cases
    return np.minimum(4.0, 4.0*(1.0 - adiff))

def boxArea(box):        
    d1 = pdist(box[0],box[1])
    d2 = pdist(box[1],box[2])
    return d1*d2
    
# boxiness measure:   contour length == rectangle perimiter?
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

#  How many corners have contour extending along the sides beyond corner?
#    (e.g. visual 90 deg angles in the contour)
def boxyCornersSides(blob, box, image):
    ct = blob  # the contourr
    lcon = len(ct)
    #print ('contour: ', ct)
    box_side_len_pts = 10  # contour pts either side of corner
    box_side_distmax = 5  # px distance
    nsidecorners = 0
    dmins, cidxs = boxyCornersList(blob, box)
    bi = 0
    for i in cidxs: # go through the box's corner points 
        cv2.circle(image, box[bi], 8, bpar.colors['blue'], 1)
        bi+= 1
        #print('\n\n\nNew corner ...')
        if dmins[i] < bpar.corner_dist_max_px:
            #get the line points for this corner
            clen = blob.shape[0]
            lp0 = ct[i][0] #the corner point
            lp1 = ct[(i+1)%clen][0]  #ahead point
            c2idx = i-1
            if c2idx < 0:  # wrap around the box
                c2idx =  clen -1  
            lp2 = ct[c2idx]    # behind point
            
            # this box corner is close to contour
            # study neighbor points on contour to this corner
            sidecorner = True
            dTotal = 0.0
            ntotal = 0
            for j in range(box_side_len_pts): 
                i2 = (i-j-1)%lcon  # one side of the corner close point
                #print('\nchecking0: ',  lp0 )
                #print('checking1: ',  lp2)
                #print('i2: ', i2, type(i2))
                #print('checking3: ',  ct[i2])
                #print('checking4: ',  dpt2line(lp0,lp2,ct[i2][0]))
                dTotal += dpt2line(lp0,lp2,ct[i2][0])
                ntotal += 1
                #  below is wrong - need distance to LINE not point!!
                #if (pdist(ct[i], box(i)) < box_side_distmax and pdist(ct[j],box[i] < box_side_distmax):
                    #sidecorner = False
            for j in range(box_side_len_pts):
                i1 =   (i+j)%lcon    #wrap around the closed contour
                #print('checking5: ',  lp0 )
                #print('checking6: ',  lp2)
                #print('i1: ', i1, type(i1))
                #print('checking8: ',  ct[i1])
                #print('checking9: ',  dpt2line(lp0,lp2,ct[i1][0]))
                dTotal += dpt2line(lp0,lp2,ct[i1][0])
                ntotal += 1
            #print('avg dist: {:5.1f}'.format( dTotal/ntotal))
            if dTotal/ntotal <= box_side_distmax:
                cv2.circle(image, lp0, 5, bpar.colors['white'], 1)
                cv2.circle(image, lp1, 5, bpar.colors['white'], 1)
                cv2.circle(image, lp2, 5, bpar.colors['white'], 1)
                nsidecorners += 1
    #print('returning: ', nsidecorners, ' side corners')
    return nsidecorners        
                
def dpt2line(p1, p2, pt):
    x0 = pt[0]
    y0 = pt[1]
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    num = np.abs((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1))
    den = np.sqrt((x2-x1)**2+(y2-y1)**2)
    return num/den

##   How many box corners are close to a single contour pt. 
def boxyCorners(blob, box):
    dmins, idxs = boxyCornersList(blob,box)
    nclose = 0
    for i in range(4):
        if dmins[i] < bpar.corner_dist_max_px:
            nclose += 1   # how many corners are close to contour
    return nclose

def boxyCornersOrig(blob, box):
    dmins, idxs = boxyCornersList(blob,box)
    nclose = 0
    for i in range(4):
        if dmins[i] < bpar.corner_dist_max_px:
            nclose += 1   # how many corners are close to contour
    return nclose

# basic computations for boxiness measures
def boxyCornersList(blob, box):
    # Find closest points to box corners in the contour 
    dmins = np.zeros(4) 
    dminContIdx = [int(0),int(0),int(0),int(0)]
    for i,d in enumerate( dmins ):
        dmins[i] = 999999999    # huge value to pass 'di <' below
        
    for i, corner in enumerate(box):
        for j,contPt in enumerate(blob[0]):
            print('\nboxy: ',corner, contPt, j)
            print('boxy: ',corner.shape, contPt.shape)
            print('boxy: blob shape: ', blob.shape, blob[0].shape)
            di = pdist(contPt,corner)
            if di < dmins[i]:
                dmins[i] = di  # closest blob point to corner
                dminContIdx[i] = int(j)  # index of closest blob point in contour  
    return dmins, dminContIdx

            

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
