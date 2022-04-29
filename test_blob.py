import numpy as np
import cv2
import sys as sys

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
        selection = np.where(img == color, img, 0)

        # split image and selection by channels as next code doesn't work
        #   with multichannel images
        channels_img = cv2.split(img)
        channels_sel = cv2.split(selection)
        csel = []
        cimg = []
        print('Shape: channels_sel: ', np.shape(channels_sel))
        print('Shape: channels_sel[1]: ',  np.shape(channels_sel[1]))
        print('Type:  channels_sel[1]: ', type( channels_sel[1]))
        for i in range(len(channels_sel)):
            # remove noise pixels of the same color 
            
            #channels_sel[i] = cv2.erode(channels_sel[i], erode_kernel)
            x = cv2.erode(channels_sel[i], erode_kernel) 
            #channels_sel[i] = x
            csel.append(x)
            
            # now expand selected blob
            # note that dilation kernel must compensate erosion so 
            #   add erosion kernel size to it
            csel[i] = cv2.dilate(csel[i], dilate_kernel)
            x = cv2.dilate(csel[i], dilate_kernel)
            cimg.append(x)
            
            # replace fragment on original image with expanded blob
            mask = cv2.threshold(csel[i], 0, 255, cv2.THRESH_BINARY_INV)[1]
            cimg[i] = cv2.bitwise_and(cimg[i], mask)
            cimg[i] = cv2.bitwise_or(cimg[i], csel[i])

        # merge processed channels back
        imgProc = cv2.merge(cimg)
        selection = cv2.merge(csel)
        cv2.imshow('processed selection', imgProc)
        #cv2.imshow('selection', selection)


name = sys.argv[1]

img = cv2.imread(name)
cv2.imshow(winname, img)
cv2.setMouseCallback(winname, on_mouse, img)
cv2.waitKey()
