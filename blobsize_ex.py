import cv2
import numpy as np
from PIL import Image

import sys

def blobize(im, blob_params):
    '''
    Takes an image and tries to find blobs on this image. 

https://stackoverflow.com/questions/65026468/cv2-simpleblobdetector-difficulties

    Parameters
    ----------
    im : nd.array, single colored. In case of several colors, the first
    color channel is taken. Alternatively you can provide an 
    PIL.Image, which is then converted to "L" and used directly.
    
    blob_params : a cv2.SimpleBlobDetector_Params() parameter list

    Returns
    -------
    blobbed_im : A greyscale image with the found blobs circled in red
    keypoints : an OpenCV keypoint list.

    '''
    if Image.isImageType(im):
        im = np.array(im.convert("L"))
    if isinstance(im, np.ndarray):
        if (len(im.shape) >= 3 
        and im.shape[2] > 1):
            im = im[:,:,0]
            
    detector = cv2.SimpleBlobDetector_create(blob_params)
    try:
        keypoints = detector.detect(im)
    except:
        keypoints = None
    if keypoints:    
        blobbed_im = cv2.drawKeypoints(im, keypoints, np.array([]), 
                    (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        blobbed_im = im
    return blobbed_im, keypoints



if __name__== '__main__':
    if len(sys.argv) != 2:
        print ('supply an arg')
        quit()
    im1 = cv2.imread(sys.argv[1])
    #im2 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = im1
    
    blob_params = cv2.SimpleBlobDetector_Params()

    # images are converted to many binary b/w layers. Then 0 searches for dark blobs, 255 searches for bright blobs. Or you set the filter to "false", then it finds bright and dark blobs, both.
    blob_params.filterByColor = False
    blob_params.blobColor = 0 

    # Extracted blobs have an area between minArea (inclusive) and maxArea (exclusive).
    blob_params.filterByArea = True
    blob_params.minArea = 250. # Highly depending on image resolution and dice size
    blob_params.maxArea = 10000. # float! Highly depending on image resolution.

    blob_params.filterByCircularity = True
    blob_params.minCircularity = 0.7 # 0.7 could be rectangular, too. 1 is round. Not set because the dots are not always round when they are damaged, for example.
    blob_params.maxCircularity = 3.4028234663852886e+38 # infinity.

    blob_params.filterByConvexity = False
    blob_params.minConvexity = 0.
    blob_params.maxConvexity = 3.4028234663852886e+38

    blob_params.filterByInertia = True # a second way to find round blobs.
    blob_params.minInertiaRatio = 0.55 # 1 is round, 0 is anywhat 
    blob_params.maxInertiaRatio = 2.00 # infinity again

    blob_params.minThreshold = 0 # from where to start filtering the image
    blob_params.maxThreshold = 255.0 # where to end filtering the image
    blob_params.thresholdStep = 5 # steps to go through
    blob_params.minDistBetweenBlobs = 3.0 # avoid overlapping blobs. must be bigger than 0. Highly depending on image resolution! 
    blob_params.minRepeatability = 2 # if the same blob center is found at different threshold values (within a minDistBetweenBlobs), then it (basically) increases a counter for that blob. if the counter for each blob is >= minRepeatability, then it's a stable blob, and produces a KeyPoint, otherwise the blob is discarded.

    res, keypoints = blobize(im2, blob_params)
    
    cv2.imshow('title',res)
    cv2.waitKey(0)
    
'''
  params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 100;
        params.maxThreshold = 256;

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 1500

        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.001

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv2.SimpleBlobDetector(params)
        else : 
            detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(clusterBlob)
        
        print('Number of keypoints detected (blobs): ', len(keypoints))
        print('Example: ', keypoints[0])

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        ClKP_with_keypoints = cv2.drawKeypoints(clusterBlob, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show keypoints
        cv2.imshow("Keypoints", ClKP_with_keypoints)
        '''
