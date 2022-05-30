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
            
            [ labelColorImg,  LabelImage, VQ_color_ctrs, color_dist ] = nf.KM(img2.image, N)
            
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
            
            if not os.path.isfile(bpar.tmp_img_path + nameTemplate.format(00, fn_root)):  # unless they exist
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
        bglabel = nf.Check_background(LabelImage)
        
        print('------------\n      Background: ',bglabel,'\n------------\n')
        
        fn_root = imagefilename.split('.')[0]
        nameTemplate = 'blobSet{:02d}_{:}.png'
            
        # set up morph kernels
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,  (bpar.esize, bpar.esize))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bpar.dsize, bpar.dsize))
        
        ncontours = 0
        bookcs = []
        boxycs = []
        rejectcs = []
        rawcs = []
        for i in range(bpar.KM_Clusters):
            # skip the background color
            if ( i == bglabel):
                print('\n\n              Skipping BG color: ',bglabel,'\n\n')
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
                
                #  4) mask the original image (im1Image)
                
                #print('***************************8 testing')
                #print(im1Image.ishape())
                #print(im1Image)
                
                
                i2 = bc.bookImage(im1Image.maskBin(b3), im1Image.scale)
                
                pimtype(i2)
                #  5) get contours
                origcontours, boxycontours, rejects =  t2.getBookBlobs(i2)
                
                for b in origcontours:
                    bookcs.append(b)
                for b in boxycontours:
                    boxycs.append(b)
                for r in rejects:
                    rejectcs.append(r) 
                
        
        print ('\n\n  Found {:} book contours'.format(len(bookcs)))
        print ('\n\n  Found {:} boxy contours \n'.format(len(boxycs)))
        print ('  Found {:} reject contours (A>20)\n\n'.format(len(rejectcs)))
        
        col = bpar.colors['red']
        blank = cv2.imread('tcolor.png')  # copy of the scaled image for display
        blank2 = blank.copy()             #copy for the rejects
        
        
        # Draw the books:
        for b in bookcs:   # rough books
            cv2.drawContours(blank, b.contour, -1, col, 3)
            cv2.drawContours(blank, [ b.box ], -1, bpar.colors['blue'], 3)
            boxystr = '{:3.1f}'.format(b.boxiness)
            blank = cv2.putText(blank, boxystr, b.centerpoint, bpar.font, 0.7 , bpar.colors['maroon'], 2, cv2.LINE_AA)

        # draw the accepted boxy contours
        for b in boxycs:
            #print('Drawing boxy contour: ', b)
            #print('Drawing boxy contour.box: ', b.box)
            boxystr = '{:3.1f}'.format(b.boxiness)
            blank = cv2.putText(blank, boxystr, b.centerpoint, bpar.font, 0.7 , bpar.colors['yellow'], 2, cv2.LINE_AA)
            cv2.drawContours(blank, b.contour, -1, bpar.colors['green'],2)
            cv2.drawContours(blank, [b.box]  , -1, bpar.colors['blue'],  thickness=1)

        cv2.imshow(filename,blank)
        
        for r in rejectcs:
            if r.area > 200 and r.boxiness >= bpar.enough_corners:
                #print('Found a reject boxy contour for drawing:')
                #print(r)
                cv2.drawContours(blank2, [r.box]  , -1, bpar.colors['blue'],  thickness=2) 
            else:
                cv2.drawContours(blank2, [r.box] , -1 , bpar.colors['maroon'], thickness=2)
            
            boxystr = '{:3.1f}'.format(r.boxiness)
            blank2 = cv2.putText(blank2, boxystr, r.centerpoint, bpar.font, 0.7 , bpar.colors['maroon'], 2, cv2.LINE_AA)
            cv2.drawContours(blank2, r.contour, -1, bpar.colors['green'], thickness=1)
            #cv2.drawContours(blank2, r.contour, -1, bpar.colors['green'], thickness=cv2.FILLED)
            
        cv2.imshow('rejects', blank2)
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
    a = sys.argv
    clears = []
    for i,ar in enumerate(a):
        if i >0:
            clears.append(int(ar))
    print ('Clearing: ', clears)
    if len(clears)>0:
        clearpicks(clears)
# read in image and perform VQ

    filename = 'target.jpg'     # orig devel image    
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
    
    
