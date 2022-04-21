
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
        img_orig_sm.blur_rad_mm(bpar.blur_rad_mm) 
    
    img2 = img_orig.icopy()  # no blur 
    img2 = img_orig_sm.icopy()  # no blur 
    
    ############
    #
    #  Use KMeans to posterize to N color labels
    #
    N = bpar.KM_Clusters
    img0, tmp, ctrs, color_dist = nf.KM(img2.image,N)   
    labelColorImg,  LabelImage, ctrs, color_dist = nf.KM(img2.image,N)   
    label_img = bc.bookImage(LabelImage,img2.scale)
    lcolor_img = bc.bookImage(labelColorImg, img2.scale)
    #colors_i2  = img2.icopy()

    print('label_img.image shape: {}'.format(np.shape(label_img.image)))
    print('label_img.ishape():    {}'.format(label_img.ishape()))
    print('label_img.scale:       {}'.format(label_img.scale))
    print('labelColorImg shape:   {}'.format(np.shape(labelColorImg)))
    
    #x = input('ENTER')
    
    #cv2.imshow('labels',img0)
    #cv2.waitKey(3000)
    
    imct = nf.Gen_cluster_colors(ctrs)
    if True:
        cv2.imshow("cluster colors (10 max)",imct)
        cv2.waitKey(1000)
    nfound = 0
    
    
    sh2 = label_img.ishape()   # new class uses ishape()
    print('Labeled:  {:} rows, {:} cols.'.format(sh2[0],sh2[1])) 
    
    #
    #   Find background label
    #
    backgnd = nf.Check_background(label_img.image)    
    
    
    xvals = []
    ybiasvals = []
    scores = []
    
    for x in range(-100,150,6):
        for y in range(-30,30,6):
        
            
            
            ###################################################################   Test Line 
            # make up a line       y = m0*x + b0
            
            xintercept = 80 #mm        
            ybias_mm = -12
            th = 145
            # get line params
            ld = nf.Get_line_params(label_img, th, x, bpar.book_edge_line_length_mm , y,  bpar.slice_width)  #llen=80, w=10
            
            # get the score
            lscore = nf.Get_line_score(label_img, bpar.slice_width, ld, color_dist, lcolor_img ) # x=0, th=125deg
            
            xvals.append(x)
            ybiasvals.append(y)
            scores.append(lscore)
        
    xmin = xvals[np.argmin(scores)]
    ymin = ybiasvals[np.argmin(scores)]
    smin = np.min(scores)
        
    ## rank the lines by score
    #pairs = zip(xvals,scores)
    #lpairs = list(pairs)
    #lpairs = sorted(lpairs, key = lambda x: x[1])
    
    #print('Ranked Book Report: ')
    #for p in lpairs:
        #x = p[0]
        #s = p[1]
        #l = '*'*int(20*min(1.0,s))            
        #print('X: {:6.2f}  score: {:6.3f}, {:}'.format(x,s, l))
    #line_disp_image = lcolor_img.icopy()
    line_disp_image = img_orig.icopy()
    
    ld = nf.Get_line_params(line_disp_image, th, xmin, bpar.book_edge_line_length_mm , ybias_mm,  bpar.slice_width)  #llen=80, w=10

    print('best book found: x:{:5.2f}  y:{:5.2f} score{:5.2f}'.format(xmin,ymin,smin)) 
    
    
    print('Book Report: ')
    for i,s in enumerate(scores):
        l = '*'*int(20*min(1.0,s))
        x = xvals[i]
        y = ybiasvals[i]
        print('X: {:6.2f} Y: {:6.2f}  score: {:6.3f}, {:}'.format(x,y,s, l))
        
        
        
    #
    #    Find the local minima
    #       lowest score is best
    
    #sminima = []
    #xlocs = []
    #smin1 = 100.0
    #xmin1 = x
    #peak = True    # a set of high values between minima
    #for i,x in enumerate(xvals):
        #if scores[i] < 1.0:
            #peak = False
            #if scores[i] < smin1:
                #smin1 = scores[i]
                #xmin1 = x
        #else:
            #if not peak:             
                #sminima.append(smin1)
                #xlocs.append(xmin1)
                #peak = True
            #smin1 =100.0   # reset for next local min
    #if not peak:  # if we ended NOT in a peak
        #sminima.append(smin1)
        #xlocs.append(xmin1)
        
        
    #print('Local minima report: ')
    ## draw lines for each local min
    #for i,x in enumerate(xlocs):
        #l = '*'*int(20*min(1.0,sminima[i]))
            
        #print('X: {:6.2f}  score: {:6.3f}, {:}'.format(x,sminima[i],l))
        
        
    MAX_LINE_SCORE = 0.240
    
    for i,s in enumerate(scores):
        #
        #   Draw the testing lines and bounds of best book locs 
        # 
        if s < MAX_LINE_SCORE:
            l = '*'*int(20*min(1.0,s))
            x = xvals[i]
            y = ybiasvals[i]
            ld = nf.Get_line_params(line_disp_image, th, x, bpar.book_edge_line_length_mm , y,  bpar.slice_width)  #llen=80, w=10

            colcode = nf.score2color(smin)
            if colcode == None:
                print('None is the color!')
                colcode = 'white'

            # new line draw feature of bookImage class!
            line_disp_image.Dline_ld(ld, colcode)
            
            # draw window above and below the line:
            ld['ybias'] += ld['rV']
            line_disp_image.Dline_ld(ld, 'blue')    
            ld['ybias'] -= 2*ld['rV']
            line_disp_image.Dline_ld(ld, 'green')
            ld['ybias'] += ld['rV']
            
            # draw red square at center of line
            line_disp_image.Dmark_mm((ld['xintercept'],ld['ybias']),3,'red')
            
            
    sh = line_disp_image.ishape()
    print('Output(line_disp_image):  {:} rows, {:} cols.'.format(sh[0],sh[1]))    
  
    ###################################################################3
    #
    #  Draw some debugging graphics          TODO:   move this inside bookImage()
    #
    line_disp_image.Dxy_axes()
        
        
    #draw the "ybias_mm" line2
    (xmin, xmax, ymin, ymax) = line_disp_image.Get_mmBounds()
    ## Draw the effective midpoint in Y (row_bias_mm)
    line_disp_image.Dline_mm((xmin+20,ybias_mm), (xmax-20,ybias_mm), 'red')


    title='test image'
    cv2.imshow(title, line_disp_image.image)
    cv2.waitKey(0) 
