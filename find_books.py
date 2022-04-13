
import cv2
import numpy as np
import time as time
import glob as gb
#from book_hough import *
import newfcns as nf
import book_classes as bc
import book_parms as bpar


    
#img_paths = gb.glob('tiny/testimage2.png')
img_paths = gb.glob('tiny/target.jpg')
d2r = 2*np.pi/360.0  #deg to rad

if (len(img_paths) < 1):
    print('No files found')
    quit()
for pic_filename in img_paths:
    print('looking at '+pic_filename)
    #img_gray, img = pre_process(pic_filename)
    #cv2.IMREAD_GRAYSCALE
    
    #
    
    #
    #  read in the image
    #
    #img = cv2.imread(pic_filename, cv2.IMREAD_COLOR)
    #
    #   nominal scale of the real images is 5mm/pixel = 0.2pixels/mm
    book_shelf = bc.bookImage(cv2.imread(pic_filename, cv2.IMREAD_COLOR), 0.2) 
    tsti = book_shelf.copy()  # copy of original for visualization 

    sh = book_shelf.ishape()
    print('Original:   {:} rows, {:} cols.'.format(sh[0],sh[1]))
    
    #cv2.imshow('test',book_shelf.image)
    #cv2.waitKey(1500)
    #
    #
    #  scale the image 
    #
    #     scale factor imported from newfcns.py
    #
    
    img1 = book_shelf.downvert(bpar.scale)
         
    sh = img1.ishape()
    print('Scaled:    {:} rows, {:} cols.'.format(sh[0],sh[1]))        
        
    ############
    #
    #   blur  
    #
    
    b = int(bpar.blur_rad/bpar.scale)
    if b%2 == 0:
        b+=1
        
    tmp = cv2.GaussianBlur(img1.image, (b,b), 0)
    img2 = bc.bookImage(tmp,img1.scale)


    ############
    #
    #  Use KMeans to posterize to N color labels
    #
    N = bpar.KM_Clusters
    img0, tmp, ctrs, color_dist = nf.KM(img2.image,N)   
    label_img = bc.bookImage(tmp,img2.scale)
    print('label_img.image shape: {}'.format(np.shape(label_img.image)))
    cv2.imshow("labeled KM image", img0)
    
    
    #
    #  Display a card of the color clusters
    #
    imct = nf.Gen_cluster_colors(ctrs)
    if True:
        cv2.imshow("cluster colors (10 max)",imct)
        cv2.waitKey(1000)
    
    
    sh = label_img.ishape()   # new class uses ishape()  == self.image.shape
    print('Labeled Image:  {:} rows, {:} cols.'.format(sh[0],sh[1])) 
    
    #
    #   Find background label
    #
    backgnd = nf.Check_background(label_img.image)    
    
    #
    #  Generate a bunch of lines all over the image and look for uniform color on either side
    #
    
    scores = []
    lines  = []
    
    scanstep = 25
    for xintercept in range(-80,80,scanstep):
        for th in range(120, 160, 5):   # line angle
            for ybias_mm in range(-100,25,scanstep):
                book_shelf.DMark_mm((xintercept, ybias_mm), 2.0, 'red')
                #get the line parameters in form of a dictionary
                ld = nf.Get_line_params(th, xintercept, bpar.book_edge_line_length_mm , ybias_mm, bpar.slice_width) 
                # measure how booklike is this line (lower score is better)
                lscore = nf.Get_line_score(label_img, bpar.slice_width, ld, color_dist)  # x=0, th=125deg
                #if lscore < 0.35:   # magic number!!
                scores.append(lscore)
                lines.append(ld)
                    
                    
    # we're done sesarching
    # display high scoring lines 
    for i,ld in enumerate( lines):
        color = nf.score2color(scores[i])
        if color is not None:
            book_shelf.Dline_ld(ld,color)
            print('X: {} th: {} score: {}'.format(ld['xintercept'], ld['th'], scores[i]))      
        
    cv2.imshow('identified book lines', book_shelf.image)
    #cv2.imshow('tmp2', label_img.image)
    cv2.waitKey(0)
        
