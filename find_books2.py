
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
    
    #
    #  Check for stored KMeans Result and compute if not available
    #
    #
    N = bpar.KM_Clusters  # how many VQ clusters to generate
    
    [ labelColorImg,  LabelImage, VQ_color_ctrs, color_dist ] = nf.VQ_pickle(img2.image, N)
    label_img  = bc.bookImage(LabelImage,img2.scale)
    lcolor_img = bc.bookImage(labelColorImg, img2.scale)
    print('Found ',len(VQ_color_ctrs), ' Color Clusters')
    lcolor_img.write()
    #quit()
        
    # if redo VQ mode(!)
    #quit()  ########################3  temp
    
    ###################   write out binary blob images
    if False:
        for i in range(len(VQ_color_ctrs)):
            #  write out a binary image for each VQ color cluster 
            #2) output binary image label_img==i)
            print('VQ cluster color:', i, VQ_color_ctrs[i])
            tcol = VQ_color_ctrs[i]
            print('VQ cluster color:', i, tcol)
            levelIblobs = np.where(lcolor_img.image == tcol , lcolor_img.image, 0)
            name = 'blobSet{:02d}.png'.format(i)
            cv2.imwrite(name, levelIblobs)
        quit()
    
    
    ##   Generate an image showing the cluster colors as a palette
    if True:
        imct = nf.Gen_cluster_colors(VQ_color_ctrs)
        cv2.imshow("cluster colors ",imct)
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
                        lscore, deets = nf.Get_line_score(label_img, bpar.slice_width, ld, color_dist, lcolor_img ) # x=0, th=125deg
                        
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
        stmp = '[ '  # make a fake string printout 
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
    print('Searching cluster neighborhoods ...')
    for c in book_clustersXYTH:
        x = c[0]
        y = c[1]
        th = c[2]
        ld = nf.Get_line_params(label_img, th, x, bpar.book_edge_line_length_mm , y,  bpar.slice_width)  #llen=80, w=10
        clscoremin, scdeets = nf.Get_line_score(label_img, bpar.slice_width, ld, color_dist, lcolor_img )    # img, w, ld, cdist, testimage
        ldmin = ld
        for dx in range(-r,r,2):   # mm neighborhood
            for dy in range(-r,r,2):  # mm neighborhood
                for dth in range(-int(ang/2),int(ang/2),3):  # theta neighborhood
                    ld = nf.Get_line_params(label_img, th+dth, x+dx, bpar.book_edge_line_length_mm , y+dy,  bpar.slice_width)  #llen=80, w=10
                    sc, scdeets = nf.Get_line_score(label_img, bpar.slice_width, ld, color_dist, lcolor_img)
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
    blobN = 0

    for c in superclusters:        
        x = c['xintercept']
        y = c['ybias'] 
        r = 2 #mm
        scscore, scsdeets = nf.Get_line_score(label_img, bpar.slice_width, c, color_dist, lcolor_img)
        print('Supercluster: ({:4.1f},{:4.2f}), score: {:5.2f}'.format(x,y,scscore))
        print('    dom label: [', end='')
        for d in scsdeets:
            print(' {:5.2f}, '.format(d),end='')
        print(']')
          
#############################################################
#
#   Display the supercluster blobs
#
# 
        img = lcolor_img.image
 
        winname = 'interactive color blobs'
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,  (bpar.esize, bpar.esize))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bpar.dsize, bpar.dsize))
        
         # get only pixels of selected color with black background
        
        color = VQ_color_ctrs[scsdeets[0]]   # BGR color of dominant label in the supercluster
        
        print('Looking at color: ', color)
        
        
        clusterBlob = np.where(img == color, img, 0)


        ##  Initial approach
        if False:
            # split image and clusterBlob by channels as next code doesn't work
            #   with multichannel images
            channels_img = cv2.split(img)
            clusterBlobChans = cv2.split(clusterBlob)
            clusterBlobChannels = []
            cimg = []
            for i in range(len(clusterBlobChans)):
                # remove noise pixels of the same color 
                
                x = cv2.erode(clusterBlobChans[i], erode_kernel) 
                clusterBlobChannels.append(x)
                
                # now expand selected blob
                # note that dilation kernel must compensate erosion so 
                #   add erosion kernel size to it
                clusterBlobChannels[i] = cv2.dilate(clusterBlobChannels[i], dilate_kernel)
                x = cv2.dilate(clusterBlobChannels[i], dilate_kernel)
                cimg.append(x)
                
                # replace fragment on original image with expanded blob
                mask = cv2.threshold(clusterBlobChannels[i], 0, 255, cv2.THRESH_BINARY_INV)[1]
                cimg[i] = cv2.bitwise_and(cimg[i], mask)
                cimg[i] = cv2.bitwise_or(cimg[i], clusterBlobChannels[i])

            # merge processed channels back
            imgProc = cv2.merge(cimg)
            clusterBlob = cv2.merge(clusterBlobChannels)
            
        # streamlined approach:  work with the labelimage
        if True:
            #_,binary = cv2.threshold(LabelImage.image,100,255,cv2.THRESH_BINARY)
            label = 12
            #mask = label_img.image == label
            #binaryImg = label_img.image.copy()[mask]
            print('label_img stats:', label_img.ishape(), label_img.image.shape)
            col = VQ_color_ctrs[label]
            clusterBlob = np.where(labelColorImg==col, col, bpar.colors['black']).astype("uint8")
            #clusterBlob = np.logical_and(labelColorImg, binaryImg)
            print('clusterBlob stats:',  clusterBlob.shape)

         #im = imC.image
        ##blank = imC.blank()
        #blank = cv2.imread('tcolor.png')
        #gimg = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #ret, bimg = cv2.threshold(gimg,127,255,cv2.THRESH_BINARY)  
        #contours, hierarchy = cv2.findContours(bimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        
        
        cv2.imshow('processed clusterBlob', clusterBlob)
        cv2.waitKey(-1)
        
        cv2.imwrite('blobSet'+str(blobN)+'.png',clusterBlob)
        blobN += 1
        
        ###
        #
        #  find the blobs
        #
                # Setup SimpleBlobDetector parameters.
      
            
        cv2.waitKey(3000)
                
                
#############################################################
#
##  evaluate (somehow) book ID quality 
#
#


    title='test image'
    cv2.imshow(title, line_disp_image.image)
    cv2.waitKey(0) 

