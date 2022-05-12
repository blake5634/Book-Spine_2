#$!/usr/bin/python
#
#    Pipeline for finding books featuring result caching
#
  
import numpy as np
import cv2
    
    
#  book finding functions from this project
import book_classes as bc 
import book_parms as bpar 
import newfcns as nf
import t2blob as t2

import time
import sys
import os as os
import glob as gb
import pickle     # for storing data btwn steps

#import glob as gb
import matplotlib.pyplot as plt

def pimtype(img):
    if type(img) != type( np.ndarray([5,5])):
        print('Image type: ', type(img))
        print(img.ishape())
    else:        
        print('Image type: ', type(img))
        print(img.shape)
    return

# where the pickles will live
pickle_dir = 'VQ_pickle/'

def step00():
    i = 0
    print('Starting Step ', i)
    nm = 'step{:02d}'.format(i)
    if pick_exists(nm):
        pl = readpick(nm)
    else:
        #############################################################
        #
        #  read and pre-process image
        #
        #
        # can use '*' to find a bunch of files
        img_paths = gb.glob('tiny/target.jpg')       # actual
        d2r = 2*np.pi/360.0  #deg to rad

        if (len(img_paths) < 1):
            print('No files found')
            quit()
        for pic_filename in img_paths:
            print('looking at '+pic_filename)
            
            #
            #  read in the image
            #
            img_orig = bc.bookImage(cv2.imread(pic_filename, cv2.IMREAD_COLOR), 0.2) 
            #line_disp_image = img_orig.icopy()  # copy of original for visualization 

            sh = img_orig.ishape()
            print('Original:   {:} rows, {:} cols.'.format(sh[0],sh[1]))
            
            #
            #
            #  scale the image 
            #
            #     scale factor imported from newfcns.py
            #
            
            img_orig_sm = img_orig.downvert(2)     # scale down rows and cols by 2
                
            sh = img_orig_sm.ishape()
            print('Scaled:    {:} rows, {:} cols.'.format(sh[0],sh[1]))        
                
            ############
            #
            #   blur  
            #
            
            blur = False
            
            if blur:
                print('Starting image blur.')
                img_orig_sm.blur_rad_mm(bpar.blur_rad_mm) 
            
            img2 = img_orig.icopy()  # no blur 
            img2 = img_orig_sm.icopy()  # no blur 
            

            #############################################################
            #
            #  VQ the colors 
            #
            # 
            print('Starting colors VQ')
            N = bpar.KM_Clusters  # how many VQ clusters to generate
            
            [ labelColorImg,  LabelImage, VQ_color_ctrs, color_dist ] = nf.KM(img2.image, N)
            
            label_img  = bc.bookImage(LabelImage,img2.scale)
            lcolor_img = bc.bookImage(labelColorImg, img2.scale)
            print('Found ',len(VQ_color_ctrs), ' Color Clusters')
                    
                
            ################### 
            #
            #   write out binary blob images
            
            #  TODO: generalize this to multiple images under analysis
            
            
            if not os.path.isfile('blobSet00.png'):  # unless they exist
                    for i in range(len(VQ_color_ctrs)):
                        #  write out a binary image for each VQ color cluster 
                        tcol = VQ_color_ctrs[i]
                        levelIblobs = np.where(lcolor_img.image == tcol , lcolor_img.image, 0)
                        name = 'blobSet{:02d}.png'.format(i)
                        cv2.imwrite(name, levelIblobs) 
            
            ##   Generate an image showing the cluster colors as a palette
            if False:
                imct = nf.Gen_cluster_colors(VQ_color_ctrs)
                cv2.imshow("cluster colors ",imct)
                cv2.waitKey(1000)  
    

            ############
            #
            #  save the pickle payload for this step
            # 
            pick_payload = [ labelColorImg,  LabelImage, VQ_color_ctrs, color_dist , label_img, lcolor_img] 
            print(nm+' saving pickle...')
            writepick(nm, pick_payload)
                    
    
def step01():
    i = 1
    print('Starting Step ', i)
    nm1 ='step{:02d}'.format(i-1)  # result of previous step
    nm  ='step{:02d}'.format(i)    # did we do this step already?
    
    #ppl = readpick(nm1)      # payload from previous step
    #[ labelColorImg,  LabelImage, VQ_color_ctrs, color_dist , label_img, lcolor_img] 
    
    nm = 'step{:02d}'.format(i)  # do we need to compute this step???
    if pick_exists(nm):
        pl = readpick(nm)
    else:
        #############################################################
        #
        #      Visualize the found contours
        #
        
        # set up morph kernels
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,  (bpar.esize, bpar.esize))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bpar.dsize, bpar.dsize))
        
        bookcs = []
        rejectcs = []
        rawcs = []
        for i in range(20):
            fname = 'blobSet{:02d}.png'.format(int(i))
            print('Opening: ', fname)
            
            im1 = cv2.imread(fname) 
            im1Image = bc.bookImage(im1,bpar.scale)
            
            #  1) convert to gray
            
            timg = bc.bookImage(im1Image.gray(), im1Image.scale)
            
            #  2) threshold
            
            bimg = timg.thresh(128)
             
            #  3) smooth with dilate and erode_kernel
            b2 = cv2.erode(bimg, erode_kernel)
            b3 = cv2.dilate(b2, dilate_kernel)
            
            #  4) mask the original image (im1Image)
            i2 = bc.bookImage(im1Image.maskBin(b3), im1Image.scale)
            
            pimtype(i2)
            #  5) get contours
            bcont, rawcontours, rejects =  t2.getBookBlobs(i2)
            
            
            for b in bcont:
                bookcs.append(b)
            for r in rejects:
                rejectcs.append(r)
            for oc in rawcontours:
                rawcs.append(oc)
                
        print ('\n\n  Found {:} book contours \n'.format(len(bookcs)))
        print ('\n\n  Found {:} orig contours \n'.format(len(bookcs)))
        print ('  Found {:} reject contours (A>20)\n\n'.format(len(rawcs)))
        
        col = bpar.colors['red']
        blank = cv2.imread('tcolor.png')
        blank2 = blank.copy()
        #for b in bookcs:
        for b in rawcs:
            cv2.drawContours(blank, b, -1, col, 3)

        cv2.imshow('title',blank)
        
        for r in rejectcs:
            cv2.drawContours(blank2, r, -1, bpar.colors['green'], thickness=cv2.FILLED)
            
        cv2.imshow('rejects', blank2)
        cv2.waitKey(-1)
 
    
def step02():
    i = 2
    print('Starting Step ', i)
    nm = 'step{:02d}'.format(i)
    if pick_exists(nm):
        pl = readpick(nm)
    else:
        #
        #  This is your compute step
        #
        time.sleep(2)
        pl = [x for  x in range(20)]
        writepick(nm, pl)
        
        
    print('Final result: ', pl) 
    


#
#   Generic pickle functions
#
def pick_exists(name):
    #
    #  check for file
    #
    pfname = pickle_dir + name + '_pickle.p'
    if not os.path.isdir(pickle_dir):  # if this doesn't exist, create it.
        print('Creating a new pickle directory: ./'+pickle_dir)
        os.mkdir(pickle_dir)
    elif os.path.isfile(pfname):
        return True
    else:
        return False
    
def readpick(name):
    pfname = pickle_dir + name + '_pickle.p'
    with open(pfname, 'rb') as pick:
        print('Reading pickle: '+name)
        pick_payload = pickle.load(pick)
        pick.close()
        return pick_payload
    
def writepick(name, pick_payload):
    pfname = pickle_dir + name + '_pickle.p'
    try:
        with open(pfname,'wb') as pf:   
            pprotocol = 2
            pickle.dump(pick_payload, pf, protocol=pprotocol)
            pf.close()
    except:
        print('couldnt write pickle file')
        
        
        
        
if __name__=='__main__':
# read in image and perform VQ
    step00()
# trace contours, ID rectangles, and display
    step01()
    #step02()
    
    
