############
#
#   Test book blob class and features


import numpy as np
import cv2
import sys as sys
import t2blob as t2b
import sys

import book_classes as bc
import book_parms as bpar
print('OpenCV version: ', cv2.__version__)

# https://stackoverflow.com/questions/64021471/how-to-expand-a-particular-color-blob-with-4-pixels

winname = 'interactive color blobs'
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,  (bpar.esize, bpar.esize))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bpar.dsize, bpar.dsize))
# dilate_kernel size = (<desired expansion> + (<erode_kernel size> - 1) / 2) * 2 + 1

def on_mouse(event, x, y, flag, img):
    if event == cv2.EVENT_LBUTTONUP:
        # get only pixels of selected color with black background
        color = img[y][x]
        print('Looking at color: ', color)

if __name__=='__main__':
        #cv2.imshow('processed selection', imgProc)
        #cv2.imshow('selection', selection)

        imagefilename   = sys.argv[1] 
 
            
        # set up morph kernels
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,  (bpar.esize, bpar.esize))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bpar.dsize, bpar.dsize))
        
    
        fname = bpar.tmp_img_path + imagefilename
        print('Opening: ', fname)
        
        # read the binary image segemented by color image
        im1 = cv2.imread(fname) 
        im1Image = bc.bookImage(im1, bpar.scale)
        dimg = im1Image.icopy().image
        
        #bglabel = nf.Check_background(LabelImage)
        #print('------------\n      Background: ',bglabel,'\n------------\n')
         
        #  we need a gray scale image 
        im1Image.image = im1Image.gray()
            
        ret, bimg = cv2.threshold(im1Image.image,127,255,cv2.THRESH_BINARY)  
        #contours, hierarchy = cv2.findContours(bimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(bimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #
        #  each contour is a bookblob instance
        
        for c in contours:
            bb = t2b.bblob(c)
            
            #if bb.area > 100:
                #print(bb)
                #x = input('ENTER:')
            
            cv2.drawContours(dimg, bb.contour, -1, bpar.colors['red'], 3)

        cv2.imshow(winname, dimg)
        cv2.setMouseCallback(winname, on_mouse, dimg)
        cv2.waitKey()
