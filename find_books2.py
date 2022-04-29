
import cv2
import numpy as np
import time as time
import glob as gb
import newfcns as nf
import book_classes as bc
import book_parms as bpar
import book_classes as bc
import matplotlib.pyplot as plt


import pickle     # for storing pre-computed K-means clusters
import sys as sys
import os as os

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
        img_orig_sm.blur_rad_mm(bpar.blur_rad_mm) 
    
    img2 = img_orig.icopy()  # no blur 
    img2 = img_orig_sm.icopy()  # no blur 
    

#############################################################
#
#  VQ the colors and pickle them
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
        img_orig_sm.blur_rad_mm(bpar.blur_rad_mm) 
    
    #img2 = img_orig.icopy()  # no blur 
    img2 = img_orig_sm.icopy()  # no blur 
    
    
    
    #
    #  Check for stored KMeans Result and compute if not available
    #
    #
    N = bpar.KM_Clusters  # how many VQ clusters to generate
    
    [ labelColorImg,  LabelImage, ctrs, color_dist ] = nf.VQ_pickle(img2.image, N)
    label_img  = bc.bookImage(LabelImage,img2.scale)
    lcolor_img = bc.bookImage(labelColorImg, img2.scale)
    print('Enter name for label color image (tcolor) e.g.)')
    #lcolor_img.write()
    #label_img.write()
    
    # if redo VQ mode(!)
    #quit()  ########################3  temp
    
    
    ##   Generate an image showing the cluster colors as a palette
    if False:
        imct = nf.Gen_cluster_colors(ctrs)
        cv2.imshow("cluster colors (10 max)",imct)
        cv2.waitKey(1000)  
    

#############################################################
#
#  find lines
#
# 
    #
    #   Find background label
    #
    backgnd = nf.Check_background(label_img.image)   
    
    MAX_LINE_SCORE = bpar.Line_Score_Thresh

    ldvals = [] # redundant with th,x,y!!
    thvals = []
    xvals = []
    ybiasvals = []
    scores = []
    xy_set = set()
    
    
    #
    #  check for and load a pickle file to save time
    #
    pickle_dir = 'VQ_pickle/'
    if not os.path.isdir(pickle_dir):  # if this doesn't exist, create it.
        print('Creating a new pickle directory: ./'+pickle_dir)
        os.mkdir(pickle_dir)
    name = pickle_dir + 'line_scores' + '_pickle.p'        #print 'WRONG - quitting, error: ',sys.exc_info()[0]
    
    [ldvals, thvals, xvals, ybiasvals, scores, xy_set] = nf.linescores_pickle(name)
    
    # but if there is no pickle file:
    if thvals == None:
        # rdef these because they were set to "None" as a signal by linescores_pickle
        ldvals = [] # redundant with th,x,y!!
        thvals = []
        xvals = []
        ybiasvals = []
        scores = []
        xy_set = set()        #sys.exit 
        print(' Creating and storing line score pickle for '+ '('+name+')')
        with open(name,'wb') as pf:   
            for theta in range(120,170,3):
                for x in range(-110,140,10):
                    for y in range(-50,40,10):
                        ###################################################################   Test Line 
                        # make up a line       y = m0*x + b0
                    
                        # get line params
                        ld = nf.Get_line_params(label_img, theta, x, bpar.book_edge_line_length_mm , y,  bpar.slice_width)  #llen=80, w=10
                        
                        # get the score
                        lscore = nf.Get_line_score(label_img, bpar.slice_width, ld, color_dist, lcolor_img ) # x=0, th=125deg
                        
                        # if score is low enough store the line and score
                        if lscore < MAX_LINE_SCORE:
                            ldvals.append(ld)
                            thvals.append(theta)
                            xvals.append(x)
                            ybiasvals.append(y)
                            scores.append(lscore)
                            xy_set.add((x,y) )     # the points that have >=1 good line
            pprotocol = 2
            pick_payload = [ldvals, thvals, xvals, ybiasvals, scores, xy_set]
            pickle.dump(pick_payload, pf, protocol=pprotocol)
            pf.close()
        
    
    # image we will use to display lines
    line_disp_image = img_orig.icopy()
    

#############################################################
#
#  rank and select best lines
#
#   
    sortbuf = []
    for i,s in enumerate(scores):
        sortbuf.append([s, ldvals[i]])
    sortbuf.sort(key=lambda x: x[0],reverse=False)
    
    for i in range(bpar.topNbyscore):
        s = sortbuf[i][0]
        ld = sortbuf[i][1]
        
        #print('Line: x0: {:4.1f}  y0: {:4.1f} th: {:4.1f} sc: {:5.3f}'.format(ld['xintercept'],ld['ybias'],ld['th'],s))
        
        colcode = nf.score2color(s)
        # new line draw feature of bookImage class!
        line_disp_image.Dline_ld(ld, colcode)
        
        # draw window above and below the line:
        ld['ybias'] += ld['rV']
        line_disp_image.Dline_ld(ld, 'black', 1)    
        ld['ybias'] -= 2*ld['rV']
        line_disp_image.Dline_ld(ld, 'black', 1)
        ld['ybias'] += ld['rV']
        
        # draw red square at center of line
        line_disp_image.Dmark_mm((ld['xintercept'],ld['ybias']),3,'red')
        

#############################################################
#
# VQ line groups 
#
#
    book_clustersXYTH, counts = nf.KM_ld(sortbuf[0:bpar.topNbyscore], bpar.line_VQ_Nclusters)
    print('total clusters:')
    for c in book_clustersXYTH:
        stmp = '[ '
        for xy in c:
            stmp += '{:5.1f},'.format(xy)
        stmp = stmp[:-1] + ']'
        print('    ',stmp)
        line_disp_image.Dmark_mm((c[0],c[1]),5,'green')
        
#############################################################
#
# Find the BEST line in the neighborhood of the line group cluster codeword 
#             (superclusters)
#
    side = bpar.KMneighborDX   # how far to search around the line cluster cw
    ang = bpar.KMneighborDth
    
    r = int(side/2.0)    
    superclusters = []
    superscores =   []
    # search in the neighborhood of each cluster for a better "cluster"
    for c in book_clustersXYTH:
        x = c[0]
        y = c[1]
        th = c[2]
        ld = nf.Get_line_params(label_img, th, x, bpar.book_edge_line_length_mm , y,  bpar.slice_width)  #llen=80, w=10
        clscoremin = nf.Get_line_score(label_img, bpar.slice_width, ld, color_dist, lcolor_img )    # img, w, ld, cdist, testimage
        ldmin = ld
        for dx in range(-r,r,2):   # mm neighborhood
            for dy in range(-r,r,2):  # mm neighborhood
                for dth in range(-int(ang/2),int(ang/2),3):  # theta neighborhood
                    ld = nf.Get_line_params(label_img, th+dth, x+dx, bpar.book_edge_line_length_mm , y+dy,  bpar.slice_width)  #llen=80, w=10
                    sc = nf.Get_line_score(label_img, bpar.slice_width, ld, color_dist, lcolor_img)
                    if  sc < clscoremin:
                        clscoremin = sc
                        ldmin = ld

        superclusters.append(ldmin)
        superscores.append(clscoremin)
        
    for c in superclusters:
        x = c['xintercept']
        y = c['ybias'] 
        line_disp_image.Dmark_mm((x,y),7,'blue')
        
        

#############################################################
#
#  identify color blobs centered around supercluster lines
#
#

    for c in superclusters:        
        x = c['xintercept']
        y = c['ybias'] 
        r = 2 #mm
        labs = set()
        scscore, scsdeets = nf.Get_line_score(label_img, bpar.slice_width, c, color_dist, lcolor_img)
        print('Supercluster: ({:4.1f},{:4.2f}), score: {:5.2f}'.format(x,y,scscore))
        print('    dom label: ', scsdeets)
                
                
#############################################################
#
##  evaluate (somehow) book ID quality 
#
#


    title='test image'
    cv2.imshow(title, line_disp_image.image)
    cv2.waitKey(0) 

