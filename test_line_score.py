
import cv2
import numpy as np
import time as time
import glob as gb
from book_hough import *
import newfcns as nf
import book_parms as bpar
import book_classes as bc
import matplotlib.pyplot as plt
'''
Test some new functions for 
Find books in a bookshelf image


 **********  look for major lines at different angles
 
 4/22   adapt to use book_classes
''' 
 

#img_paths = gb.glob('tiny/target.jpg')
img_paths = gb.glob('tiny/testimage2.png')
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
    img_orig = bookImage(cv2.imread(pic_filename, cv2.IMREAD_COLOR), 0.2) 
    tsti = img_orig.copy()  # copy of original for visualization 

    sh = img_orig.shape()
    print('Original:   {:} rows, {:} cols.'.format(sh[0],sh[1]))
    
    #
    #
    #  scale the image 
    #
    #     scale factor imported from newfcns.py
    #
    
    img1 = img_orig.downvert(bpar.scale)
        
    #img_width = scaled_ish[1]
    #img_height = scaled_ish[0] 
    #img1 = cv2.resize(img_orig, (img_width, img_height))
    
    sh = img1.shape()
    print('Scaled:    {:} rows, {:} cols.'.format(sh[0],sh[1]))        
        
    ############
    #
    #   blur  
    #
    
    b = int(bpar.blur_rad/bpar.scale)
    if b%2 == 0:
        b+=1
        
    tmp = cv2.GaussianBlur(img1, (b,b), 0)
    img2 = bookImage(tmp,bpar.scale)


    ############
    #
    #  Use KMeans to posterize to N color labels
    #
    N = bpar.KM_Clusters
    img0, label_img, ctrs, color_dist = nf.KM(img2,N)   
    #print('label_img shape: {}'.format(np.shape(label_img)))
    #cv2.imshow("labeled KM image", img0)
    
    imct = nf.Gen_cluster_colors(ctrs)
    if True:
        cv2.imshow("cluster colors (10 max)",imct)
        cv2.waitKey(1000)
    nfound = 0
    
    
    sh = label_img.shape
    print('Labeled:  {:} rows, {:} cols.'.format(sh[0],sh[1])) 
    
    #
    #   Find background label
    #
    backgnd = nf.Check_background(label_img)    
    
    #
    #  look for lines at a bunch of x-values
    #
 
    xwidth = 81
    linescanx_mm = range(xwidth,-xwidth, -2) #mm
    dth = 15 # degrees off of 135
    ang_scan_deg = range(135-dth, 135+dth,2)  # deg 
    
    lines_found = []
    number_of_all_lines = 2*xwidth*dth
    
    ###################################################################   Test Line 
    # make up a line       y = m0*x + b0
    xintercept = -5.0 #mm              
    th = 135   # deg relative to 03:00 (clock)
    ld = nf.Get_line_params(th, xintercept, bpar.book_edge_line_length_mm , bpar.slice_width)  #llen=80, w=10
    
    
    # get the score
    lscore = nf.Get_line_score(label_img, bpar.slice_width, xintercept, th, bpar.book_edge_line_length_mm, bpar.row_bias_mm, color_dist)  # x=0, th=125deg
    
    print('X: {} th: {} score: {}'.format(xintercept, th, lscore))        
     
    #
    #   Draw the testing line and bounds 
    #
  
    #print('m0: {} b0: {}mm rp:{}(pix)'.format(m0,b0,rp))
    #print('th: {} deg, iw {}  ih: {}'.format(th,iw,ih))
    dx = abs((bpar.book_edge_line_length_mm/2)*np.cos(th*d2r)) # mm
    xmin2 = xintercept - dx  #mm    X range for test line
    xmax2 = xintercept + dx  #mm
    colcode = 'yellow'
        


    #nf.DLine_mm(tsti, (xmin2, bpar.row_bias_mm + m0*xmin2+b0), (xmax2, bpar.row_bias_mm + m0*xmax2+b0), colcode,iscale=tstscale)
    ## above window line
    #nf.DLine_mm(tsti, (xmin2,  rV + m0*xmin2+b0), (xmax2,  rV + m0*xmax2+b0), 'blue',iscale=tstscale)
    #nf.DLine_mm(tsti, (xmin2, -rV + m0*xmin2+b0), (xmax2, -rV + m0*xmax2+b0), 'green',iscale=tstscale)
    
    if True:
        # Draw the line for debugging 
        x = xintercept
        dispscale = 1.0  # tsti is unscaled
        dx = abs((bpar.book_edge_line_length_mm/2)*np.cos(th*d2r))  #mm
        xa, ya = nf.XY2iXiY(tsti, x-dx, ld['m0']*(x-dx)+ld['b0'], dispscale)  # parms in mm
        xb, yb = nf.XY2iXiY(tsti, x+dx, ld['m0']*(x+dx)+ld['b0'], dispscale)
        print('Image shape: {}'.format(np.shape(tsti)))
        print('pt a: {},{},  pt b: {},{}'.format(xa,ya,xb,yb))
        ir = np.shape(tsti)[0]  # image rows
        ic = np.shape(tsti)[1]  # image cols
        
        # equation line
        cv2.line(tsti, (xa,ya), (xb,yb), (255,255,255), 3)

        # line window border lines
        r = int((bpar.scale/dispscale)*bpar.mm2pix*bpar.slice_width/np.cos((180-th)*d2r))  # pix
        ya += r
        yb += r
        cv2.line(tsti, (xa,ya), (xb,yb), (255,0,0), 2)
        ya -= 2*r
        yb -= 2*r
        cv2.line(tsti, (xa,ya), (xb,yb), (255,0,0), 2)

        #draw a test rectangle
        cv2.rectangle(tsti, (215,286), (425,496),(200,200,200), 3)
    
  
    
    sh = tsti.shape
    print('Output(tsti):  {:} rows, {:} cols.'.format(sh[0],sh[1]))    
  
    ###################################################################3
    #
    #  Draw some debugging graphics
    #
    # Draw H and V axes (X,Y axes in mm)    
    (xmin, xmax, ymin, ymax) = nf.Get_mmBounds(tsti)

    tstscale = bpar.scale
    nf.DLine_mm(tsti, (xmin,0), (xmax,0),'white',iscale=tstscale)
    nf.DLine_mm(tsti, (0, ymin), (0, ymax), 'white',iscale=tstscale)

    ## Draw some tick marks
    tick_locs_mm = [] # pix
    tickwidth = 20 * bpar.scale# mm
    for xt in range(int(xmax/tickwidth)): # unitless
        xpt = tickwidth*(xt+1)  # mm
        tick_locs_mm.append(xpt)
        tick_locs_mm.append(-xpt)
    ya = 0.0 #mm
    yb = -5.0 #mm
    for x in tick_locs_mm:
        nf.DLine_mm(tsti, (x, ya), (x,yb), 'green',iscale=tstscale)   # draw the tick marks
        
    ## Draw the effective midpoint in Y (row_bias_mm)
    nf.DLine_mm(tsti, (xmin+20,bpar.row_bias_mm), (xmax-20,bpar.row_bias_mm), 'green', iscale=tstscale)

    title='test image'
    cv2.imshow(title, tsti)
    cv2.waitKey(-1)
