#$!/usr/bin/python
#
#    Pipeline for finding books featuring result caching
#

import numpy as np
import cv2
    
    
#  book finding functions from this project
import book_classes as bc 
import book_parms as bpar 
import VQ_functions as vq
import blob_processing as bpr

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

def step00(imagedir, imagefilename):
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
        img_paths = gb.glob(imagedir+imagefilename)       # actual
        d2r = 2*np.pi/360.0  #deg to rad

        if (len(img_paths) < 1):
            print('No files found')
            quit()
        for pic_filename in img_paths:
            print('looking at '+imagedir+imagefilename)
            
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
            #   histogram equalization  
            #
            
            #print('Starting Histogram equalization.')
            #img_orig_sm.image = img_orig_sm.histoEq('value')
            
            
                 
            ############
            #
            #   blur  
            #
            
            blur = bpar.blur_flag
            
            if blur:
                print('Starting image blur.')
                img_orig_sm.blur_mm_rad(bpar.blur_rad_mm) 
            
            img2 = img_orig.icopy()  # no blur 
            img2 = img_orig_sm.icopy()  # no blur 
            

            cv2.imwrite('tcolor.png', img_orig_sm.image)  # true color image
            
            #############################################################
            #
            #  VQ the colors 
            #
            # 
            print('Starting colors VQ with ', bpar.KM_Clusters,' clusters.')
            N = bpar.KM_Clusters  # how many VQ clusters to generate
            
            [ labelColorImg,  LabelImage, VQ_color_ctrs, color_dist ] = vq.KM(img2.image, N)
            
            label_img  = bc.bookImage(LabelImage,img2.scale)    # pixel value = label num.
            
            lcolor_img = bc.bookImage(labelColorImg, img2.scale) # pixel value = label color
            print('Found ',len(VQ_color_ctrs), ' Color Clusters')
                    
            cv2.imwrite('label.png', label_img.image)
            cv2.imwrite('lcolor.png', lcolor_img.image)
            
            ################### 
            #
            #   write out binary blob images
            #
            #    (Note: saving as png has default compression that deletes some blobs 
            #        and in general changes the blob results)
            #
            
            #  TODO: generalize this to multiple images under analysis            
            
            fn_root = pic_filename.split('.')[0].split('/')[-1]
            nameTemplate = 'blobSet{:02d}_{:}.png'
            rmTemplate   = 'blobSet*.png'
            
            #if not os.path.isfile(bpar.tmp_img_path + nameTemplate.format(00, fn_root)):  # unless they exist
            if True:   # this is now "guarded" by pickle file 
                os.system('rm {:}'.format(bpar.tmp_img_path+rmTemplate))
                print('\\n\n                REMOVING '+bpar.tmp_img_path+rmTemplate+'\n\n')
                for i in range(len(VQ_color_ctrs)):
                    #  write out a binary image for each VQ color cluster 
                    tmpimg = lcolor_img.icopy()
                    #tcol = VQ_color_ctrs[i]
                    #levelIblobs = np.where(lcolor_img.image == tcol , tmpimg.image, 0)
                
                    print('label_img is:')
                    print(label_img)
                    
                    # need (r,c,3) image for np.where
                    t = cv2.cvtColor(np.float32(label_img.image), cv2.COLOR_GRAY2BGR)
                    print('t image is:')
                    print(t.shape)
                    
                    levelIblobs = np.where(t == (i,i,i), tmpimg.image, 0)
                    
                    print('LevelIblobs image is:')
                    print(levelIblobs.shape) 
                    
                                #  2) threshold
                    tim1 = bc.bookImage(levelIblobs, lcolor_img.scale)
                    # thresholding step moved here.
                    tim1.image = tim1.gray()
                    levelIblobs = tim1.thresh(2) 
                    name = bpar.tmp_img_path + nameTemplate.format(i,fn_root)
                    print('Writing: ', name, levelIblobs.shape)
                    cv2.imwrite(name, levelIblobs) 
        
            ##   Generate an image to show the cluster colors as a palette
            if True:
                imct = vq.Gen_cluster_colors(VQ_color_ctrs)
                cv2.imshow("cluster colors ",imct)
                cv2.waitKey(1000) 
    

            ############
            #
            #  save the pickle payload for this step
            # 
            pick_payload = [ labelColorImg,  LabelImage, VQ_color_ctrs, color_dist , label_img, lcolor_img] 
            print(nm+' saving pickle...')
            writepick(nm, pick_payload)
                    
    
def step01(imagedir, imagefilename):
    i = 1
    print('Starting Step ', i)
    nm1 ='step{:02d}'.format(i-1)  # result of previous step
    nm  ='step{:02d}'.format(i)    # did we do this step already?
    
    [ labelColorImg,  LabelImage, VQ_color_ctrs, color_dist , label_img, lcolor_img] = readpick(nm1)      # payload from previous step
    
    nm = 'step{:02d}'.format(i)  # do we need to compute this step???
    if pick_exists(nm):
        pl = readpick(nm)
    else:
        #############################################################
        #
        #      Process and visualize the found contours
        #
        imagefile = imagedir + imagefilename
        bglabel_top = vq.Check_background(LabelImage,'BG_TOP')
        bglabel_bot = vq.Check_background(LabelImage,'BG_BOTTOM')
        
        print('------------\n      Background: ',bglabel_top,'\n------------\n')
        print('------------\n      Background: ',bglabel_bot,'\n------------\n')
        
        fn_root = imagefilename.split('.')[0]
        nameTemplate = 'blobSet{:02d}_{:}.png'
            
        ###  Images on which to display results/ outlines
        detectedImage = cv2.imread('tcolor.png')  # copy of the scaled image for display
        rejectImage = detectedImage.copy()             #copy for the rejects
        
        # set up morph kernels
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,  (bpar.esize, bpar.esize))
        erode_kernelSm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,  (int(bpar.esize/2), int(bpar.esize/2)))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bpar.dsize, bpar.dsize))
        
        ncontours = 0
        bookcs = []
        boxycs = []
        rejectcs = []
        rawcs = []
        for i in range(bpar.KM_Clusters):
            # skip the background color
            if ( i == bglabel_top or i == bglabel_bot):
                print('\n\n              Skipping BG color: ',i,'\n\n')
                #fname = bpar.tmp_img_path + nameTemplate.format(i, fn_root)
                #cv2.imshow('BGcolor image', cv2.imread(fname))
                #cv2.waitKey(-1)
            else:
                fname = bpar.tmp_img_path + nameTemplate.format(i, fn_root)
                print('Opening: ', fname)
                
                # read the binary image segemented by color image
                im1 = cv2.imread(fname) 
                im1Image = bc.bookImage(im1,bpar.scale)
                
                #  we need a gray scale image 
                im1Image.image = im1Image.gray()
            
                
                ##  1) convert to gray  ##  update: stored imgs changed to binary
                
                #timg = bc.bookImage(im1Image.gray(), im1Image.scale)
                #bimg = timg.image
    
                
                #  3) smooth with dilate and erode_kernel
                
                bimg = im1Image.image
                b2 = cv2.erode(bimg, erode_kernel)
                b3 = cv2.dilate(b2, dilate_kernel)
                b2 = cv2.erode(bimg, erode_kernelSm)

                #  4) mask the original image (im1Image)
                
                #print('***************************8 testing')
                #print(im1Image.ishape())
                #print(im1Image)
                
                
                i2 = bc.bookImage(im1Image.maskBin(b3), im1Image.scale)
                
                pimtype(i2)
                #  5) get contours
                idstring = 'C{:02d}'.format(i)
                ntotcs, origcontours, boxycontours, rejects =  bpr.getBookBlobs(i2,rejectImage,idstring)
                
                for b in origcontours:
                    bookcs.append(b)
                for b in boxycontours:
                    boxycs.append(b)
                for r in rejects:
                    rejectcs.append(r) 
                
        
        print ('\n\n  Found {:} book contours'.format(len(bookcs)))
        print ('\n\n  Found {:} boxy contours \n'.format(len(boxycs)))
        print ('  Found {:} reject contours (A>{:})\n\n'.format(len(rejectcs),bpar.noise_area_threshold))
        print ('  Discarded {:} contours'.format(ntotcs - (len(bookcs)+len(boxycs)+len(rejectcs))))
        
        col = bpar.colors['red']
        
        def IDprint(image,b):
            ctr = (b.centerpoint[0], b.centerpoint[1]-20)  # offset up a bit
            image = cv2.putText(image, b.ID, ctr, bpar.font, 0.5, bpar.colors['green'], 2, cv2.LINE_AA)
            
        def aprint(image, b):
            # print the area on image
            astring = '{:}'.format(int(b.area))
            ctr = (b.centerpoint[0], b.centerpoint[1]+15)  # offset down a bit
            image = cv2.putText(image, astring, ctr, bpar.font, 0.5, bpar.colors['white'], 2, cv2.LINE_AA)
            
        def bprint(image, b, bx):
            ## print the numerical boxiness on image 
            #if bx > 2.0:
                #color = bpar.colors['yellow']
            #else:
                #color = bpar.colors['green']
            color = bpar.colors['yellow']

            boxystr = '{:3.1f}'.format(bx)
            image = cv2.putText(image, boxystr, b.centerpoint, bpar.font, 0.7 , color, 2, cv2.LINE_AA)
            
            
        print('Got here 1')
        # Draw the books:
        for b in bookcs:   # rough books
            cv2.drawContours(detectedImage, b.contour, -1, col, 3)
            cv2.drawContours(detectedImage, [ b.box ], -1, bpar.colors['blue'], 3)
            bprint(detectedImage, b, b.boxiness)
            aprint(detectedImage, b)            
            IDprint(detectedImage, b)
            
        # draw the smaller but boxy contours
        for b in boxycs:
            #print('Drawing boxy contour: ', b)
            #print('Drawing boxy contour.box: ', b.box)
            
            bprint(detectedImage, b, b.boxiness)
            aprint(detectedImage, b)          
            IDprint(detectedImage, b)    
            
            #boxystr = '{:3.1f}'.format(b.boxiness)
            #detectedImage = cv2.putText(detectedImage, boxystr, b.centerpoint, bpar.font, 0.7 , bpar.colors['yellow'], 2, cv2.LINE_AA)
            cv2.drawContours(detectedImage, b.contour, -1, bpar.colors['green'],2)
            cv2.drawContours(detectedImage, [b.box]  , -1, bpar.colors['green'],  thickness=1)
        
        print('Got here 2')
        #Display the contours drawn
        cv2.imshow(filename,detectedImage)
        
        for r in rejectcs:
            #if r.area > 200 and r.boxiness >= bpar.boxy_threshold:
            if r.area > bpar.area_min*0.75:
                #print('Found a reject boxy contour for drawing:')
                #print(r)
                cv2.drawContours(rejectImage, [r.box]  , -1, bpar.colors['blue'],  thickness=2) 
            else:
                cv2.drawContours(rejectImage, [r.box] , -1 , bpar.colors['green'], thickness=2)
            
            IDprint(rejectImage, r)
            #bprint(rejectImage, r, r.boxiness)
            bprint(rejectImage, r, r.boxiness)
            #bprint(rejectImage, r, bpr.boxyCornersSides(r.contour, r.box, rejectImage))
            aprint(rejectImage, r) 
            cv2.drawContours(rejectImage, r.contour, -1, bpar.colors['green'], thickness=1)
            
        #Display the 'reject' contours on another copy
        cv2.imshow('rejects', rejectImage)
        cv2.waitKey(-1)
 
    
def step02(imagedir, imagefilename):
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
        
        
def clearpicks(Ns):
    dir = 'VQ_pickle/'
    clearblobs = False
    for n in Ns:    # clear the specified pickle files
        if n==-1:
            clearblobs = True
        else:
            name = dir + 'step{:02d}_pickle.p'.format(n)
            if os.path.isfile(name):
                os.remove(name)
    if clearblobs:
        for f in gb.glob(bpar.tmp_img_path + 'blobSet*.png'):
            print('Removing:',f)
            os.remove(f)

        
        
if __name__=='__main__':
    
    
    print('\n\n                                                           - - -  Starting Run   - - - \n\n')
    a = sys.argv
    clears = []
    for i,ar in enumerate(a):
        if i >0:
            clears.append(int(ar))
    print ('Clearing: ', clears)
    if len(clears)>0:
        clearpicks(clears)
# read in image and perform VQ

    #filename = 'target.jpg'     # orig devel image    
    filename = 'FakeTest01.png'     # orig devel image    
    #filename = 'newtest01.jpg'
    #filename = 'newtest02.jpg'
    #filename = 'newtest03.jpg'
    #filename = 'newtest04.jpg'
    
    imagedir = bpar.image_dir
    
    #
    #   Here's where the magic happens!
    #    
    step00(imagedir,filename)
    step01(imagedir,filename)
    #step02()
    
    
