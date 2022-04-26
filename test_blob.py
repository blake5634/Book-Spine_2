import numpy as np
import cv2
import sys as sys

# https://stackoverflow.com/questions/64021471/how-to-expand-a-particular-color-blob-with-4-pixels

winname = 'interactive color blobs'
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
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
        for i in range(len(channels_sel)):
            # remove noise pixels of the same color
            channels_sel[i] = cv2.erode(channels_sel[i], erode_kernel)

            # now expand selected blob
            # note that dilation kernel must compensate erosion so 
            #   add erosion kernel size to it
            channels_sel[i] = cv2.dilate(channels_sel[i], dilate_kernel)

            # replace fragment on original image with expanded blob
            mask = cv2.threshold(channels_sel[i], 0, 255, cv2.THRESH_BINARY_INV)[1]
            channels_img[i] = cv2.bitwise_and(channels_img[i], mask)
            channels_img[i] = cv2.bitwise_or(channels_img[i], channels_sel[i])

        # merge processed channels back
        img = cv2.merge(channels_img)
        selection = cv2.merge(channels_sel)
        cv2.imshow(winname, img)
        cv2.imshow('selection', selection)


name = sys.argv[1]

img = cv2.imread(name)
cv2.imshow(winname, img)
cv2.setMouseCallback(winname, on_mouse, img)
cv2.waitKey()
