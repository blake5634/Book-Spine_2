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


# where the pickles will live
pickle_dir = 'VQ_pickle/'

def step00():
    i = 0
    print('Starting Step ', i)
    nm = 'step{:02d}'.format(i)
    if ifpick(nm):
        pl = readpick(nm)
    else:
        #############################################################
        #
        #  read and pre-process image
        #
        #

        img_paths = gb.glob('tiny/target.jpg')       # actual
        d2r = 2*np.pi/360.0  #deg to rad

        if (len(img_paths) < 1):
            print('No files found')
            quit()
        for pic_filename in img_paths:
            print('looking at '+pic_filename)
            #img_gray, img = pre_process(pic_filename)
            #cv2.IMREAD_GRAYSCALE
            
            #
            #  read in the image
            #
            #img = cv2.imread(pic_filename, cv2.IMREAD_COLOR)
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
            #  VQ the colors and pickle them
            #
            # 
                
            #
            #  Check for stored KMeans Result and compute if not available
            #
            #
            print('Starting colors VQ')
            N = bpar.KM_Clusters  # how many VQ clusters to generate
            
            [ labelColorImg,  LabelImage, VQ_color_ctrs, color_dist ] = nf.KM(img2.image, N)
            label_img  = bc.bookImage(LabelImage,img2.scale)
            lcolor_img = bc.bookImage(labelColorImg, img2.scale)
            print('Found ',len(VQ_color_ctrs), ' Color Clusters')
            #lcolor_img.write()
            
        
                    
            ###################   write out binary blob images
            if not os.path.isfile('blobSet00.png'):  # unless they exist
                    for i in range(len(VQ_color_ctrs)):
                        #  write out a binary image for each VQ color cluster 
                        #2) output binary image label_img==i)
                        print('VQ cluster color:', i, VQ_color_ctrs[i])
                        tcol = VQ_color_ctrs[i]
                        print('VQ cluster color:', i, tcol)
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
            #  generate the pickle payload for this step
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
    if ifpick(nm):
        pl = readpick(nm)
    else:
        #############################################################
        #
        #
        #time.sleep(2)
        #pl = [x for x in range(0,20)]
        bookcs = []
        rejectcs = []
        rawcs = []
        for i in range(20):
            fname = 'blobSet{:02d}.png'.format(int(i))
            print('Opening: ', fname)
            im1 = cv2.imread(fname) 
            im1Image = bc.bookImage(im1,bpar.scale)
            
            bcont, rawcontours, rejects =  t2.getBookBlobs(im1Image)
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
    if ifpick(nm):
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
def ifpick(name):
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
    
    
