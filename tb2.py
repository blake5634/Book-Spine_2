############
#
#   Test book blob class and features


import numpy as np
import cv2
import sys as sys
import blob_processing as bpr
import sys

import book_classes as bc
import book_parms as bpar
print('OpenCV version: ', cv2.__version__)

# https://stackoverflow.com/questions/64021471/how-to-expand-a-particular-color-blob-with-4-pixels

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

        imagefilename   = sys.argv[1]  # typically a binary image of constant label
 
            
        # set up morph kernels
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,  (bpar.esize, bpar.esize))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bpar.dsize, bpar.dsize))
        
    
        fname = bpar.tmp_img_path + imagefilename
        print('Opening: ', fname)
        # read the binary image segemented by color image
        im1 = cv2.imread(fname)
        if im1 is None:
            print("Couldn't read an image from: ",fname)
            quit()
        
        im1Image = bc.bookImage(im1, bpar.scale)
        
        # these versions of the image can be useful (written by pipe.py)
        true_col_img    = cv2.imread('tcolor.png')   #display image
        
        #cv2.imshow('true color image:', true_col_img)
        #cv2.waitKey()
        
        label_gray_img  = cv2.imread('label.png') #label image
        label_color_img = cv2.imread('lcolor.png')   #label image
        
        #bglabel = nf.Check_background(LabelImage)
        #print('------------\n      Background: ',bglabel,'\n------------\n')
         
        #  we need a gray scale image 
        im1Image.image = im1Image.gray()
        
        #cv2.imshow('grayscale image:', im1Image.image)
        #cv2.waitKey()
        
        ret, bimg = cv2.threshold(im1Image.image,127,255,cv2.THRESH_BINARY)  
        #cv2.imshow('thesholded image:', bimg)
        #cv2.waitKey()
        if False:  
            bimg = cv2.dilate(bimg, dilate_kernel)
            bing = cv2.dilate(bimg, dilate_kernel)
        
        #contours, hierarchy = cv2.findContours(bimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(bimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #
        #  each contour is a bookblob instance
        
        for c in contours:
            bb = bpr.bblob(c)
            
            #if bb.area > 100:
                #print(bb)
                #x = input('ENTER:')
            
            cv2.drawContours(true_col_img, bb.contour, -1, bpar.colors['red'], 3)
            
        winname = 'interactive color blobs'
        cv2.imshow(winname, true_col_img)
        cv2.setMouseCallback(winname, on_mouse, label_gray_img)
        cv2.waitKey()
