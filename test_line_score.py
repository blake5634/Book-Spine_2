
import cv2
import numpy as np
import time as time
import glob as gb
import newfcns as nf
import book_classes as bc
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
#img_paths = gb.glob('tiny/testimage2.png')   # simplified 
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
    tsti = img_orig.copy()  # copy of original for visualization 

    sh = img_orig.ishape()
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
    img2 = bc.bookImage(tmp,bpar.scale)


    ############
    #
    #  Use KMeans to posterize to N color labels
    #
    N = bpar.KM_Clusters
    img0, tmp, ctrs, color_dist = nf.KM(img2.image,N)   
    label_img = bc.bookImage(tmp,img2.scale)
    #print('label_img.image shape: {}'.format(np.shape(label_img.image)))
    #cv2.imshow("labeled KM image", img0)
    
    imct = nf.Gen_cluster_colors(ctrs)
    if True:
        cv2.imshow("cluster colors (10 max)",imct)
        cv2.waitKey(1000)
    nfound = 0
    
    
    sh = label_img.ishape()   # new class uses shape()
    print('Labeled:  {:} rows, {:} cols.'.format(sh[0],sh[1])) 
    
    #
    #   Find background label
    #
    backgnd = nf.Check_background(label_img.image)    
    
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
    
    xintercept = 83 #mm        
    ybias_mm = bpar.row_bias_mm
    ybias_mm = -12
    
    
    th = 145
    if True: 
    #for th in range(120, 200 , 10):
        
        # get line params
        ld = nf.Get_line_params(th, xintercept, bpar.book_edge_line_length_mm , ybias_mm,  bpar.slice_width)  #llen=80, w=10
        
        # get the score
        lscore = nf.Get_line_score(label_img, bpar.slice_width, ld, color_dist)  # x=0, th=125deg
        
        print('X: {} Y: {} th: {} score: {:5.3f}'.format(xintercept, ybias_mm, th, lscore))        
        
        #
        #   Draw the testing line and bounds 
        # 
        dx = abs((bpar.book_edge_line_length_mm/2)*np.cos(th*d2r)) # mm
        xmin2 = xintercept - dx  #mm    X range for test line
        xmax2 = xintercept + dx  #mm
        colcode = 'yellow'
        
        if lscore < 0.2:          #BEST
            colcode = 'red'
        elif lscore < 0.35:
            colcode = 'green'
        elif lscore < 0.5:
            colcode = 'white'     # OK/bad
            

        # new line draw feature of bookImage class!
        tsti.Dline_mm( (xmin2, ybias_mm + ld['m0']*xmin2+ld['b0']), (xmax2, ybias_mm + ld['m0']*xmax2+ld['b0']), colcode)
        ## above window line
        #nf.Dline_mm(tsti, (xmin2,  rV + m0*xmin2+b0), (xmax2,  rV + m0*xmax2+b0), 'blue')
        #nf.Dline_mm(tsti, (xmin2, -rV + m0*xmin2+b0), (xmax2, -rV + m0*xmax2+b0), 'green')
        
        
    sh = tsti.ishape()
    print('Output(tsti):  {:} rows, {:} cols.'.format(sh[0],sh[1]))    
  
    ###################################################################3
    #
    #  Draw some debugging graphics          TODO:   move this inside bookImage()
    #
    tsti.Dxy_axes()
        
        
    #draw the "ybias_mm" line
    (xmin, xmax, ymin, ymax) = tsti.Get_mmBounds()
    ## Draw the effective midpoint in Y (row_bias_mm)
    tsti.Dline_mm((xmin+20,ybias_mm), (xmax-20,ybias_mm), 'red')


    title='test image'
    cv2.imshow(title, tsti.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
