
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
import os as os



def VQ_pickle(image, N ):
    #
    #   Check for a pickle file of combined pre-computed Mech and Robot objects
    #
    #  TODO: refactor code to get rid of unused "test" argument
    pprotocol = 2

    pickle_dir = 'VQ_pickle/'

    if not os.path.isdir(pickle_dir):  # if this doesn't exist, create it.
        print('Creating a new pickle directory: ./'+pickle_dir)
        os.mkdir(pickle_dir)

    name = pickle_dir + 'imageVQ' + '_pickle.p'

    print(' pickle: trying to open ', name,' in ', os.getcwd())
    dumpflg = True
    pick_payload = []
    if(os.path.isfile(name)):
        with open(name, 'rb') as pick:
            dumpflg = False
            print('\Trying to read pre-computed VQ codebook and images from '+name)
            pick_payload = pickle.load(pick)
            pick.close()
            print('Successfully read pre-computed VQ codebook and image')
            print('pickle contained ', len(pick_payload[2]), ' codewords')
            if len(pick_payload[2]) != N:
                print('Pickle file has wrong image length')
                quit()
                dumpflg = True
    
    if dumpflg:
        #print 'WRONG - quitting, error: ',sys.exc_info()[0]
        #sys.exit 
        print(' Storing VQ pickle for '+ '('+name+')')
        with open(name,'wb') as pf:
                ############
                #
                #  Use KMeans to posterize to N color labels
                #
                img0, tmp, ctrs, color_dist = nf.KM(img2.image, N)   
                pick_payload = [img0, tmp, ctrs, color_dist]
                pickle.dump(pick_payload, pf, protocol=pprotocol)
                pf.close()
    
    return pick_payload



def linescores_pickle(fname):
    #
    #   Check for a pickle file of combined pre-computed Mech and Robot objects
    #
    #  TODO: refactor code to get rid of unused "test" argument
    pprotocol = 2

    print(' pickle: trying to open ', name,' in ', os.getcwd())
    dumpflg = True
    pick_payload = []
    if(os.path.isfile(fname)):
        with open(name, 'rb') as pick:
            print('\Trying to read pre-computed VQ codebook and images from '+name)
            pick_payload = pickle.load(pick)
            pick.close()
            print('Successfully read pre-computed VQ codebook and image')
            print('pickle contained ', len(pick_payload[2]), ' codewords') 
        return pick_payload
    else:
        return [name, None, None, None, None, None]  # signal no pickle file
    
    




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
    
    
    
    #
    #  Check for stored KMeans Result and compute if not available
    #
    #
    N = bpar.KM_Clusters  # how many VQ clusters to generate
    
    [ labelColorImg,  LabelImage, ctrs, color_dist ] = VQ_pickle(img2.image, N)
    label_img  = bc.bookImage(LabelImage,img2.scale)
    lcolor_img = bc.bookImage(labelColorImg, img2.scale)
    

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
    
    [ldvals, thvals, xvals, ybiasvals, scores, xy_set] = linescores_pickle(name)
    
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
    
    #ld = nf.Get_line_params(line_disp_image, theta, xmin, bpar.book_edge_line_length_mm , ybias_mm,  bpar.slice_width)  #llen=80, w=10

    #print('best book found: x:{:5.2f}  y:{:5.2f} score{:5.2f}'.format(xmin,ymin,smin)) 
    
        
    #######################################################################################3
    #
    #  sort all lines by score
    #
  
    sortbuf = []
    for i,s in enumerate(scores):
        sortbuf.append([s, ldvals[i]])
    sortbuf.sort(key=lambda x: x[0],reverse=False)
    
    for i in range(bpar.topNbyscore):
        s = sortbuf[i][0]
        ld = sortbuf[i][1]
        
        print('Line: x0: {:4.1f}  y0: {:4.1f} th: {:4.1f} sc: {:5.3f}'.format(ld['xintercept'],ld['ybias'],ld['th'],s))
        
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
        
    #############################################################################33
    #
    #   cluster the best lines
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
        
    #
    #   find the best line NEAR each KM cluster point
    #
    side = bpar.KMneighborDX
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
        
        
    #
    #  Identify book boundary / OBB surrounding the supercluster
    #
        
    imgblobs = label_img.icopy()
    mask = imgblobs.isequal(3)  #  VQ label 3
    imgblobs.image = imgblobs.image[mask]
    title = 'blobs of label 3'
    cv2.imshow(title, imgblobs.image)
    cv2.waitKey(0)
        
        
        
    sh = line_disp_image.ishape()
    print('Output(line_disp_image):  {:} rows, {:} cols.'.format(sh[0],sh[1]))    
  
    ###################################################################3
    #
    #  Draw some debugging graphics          TODO:   move this inside bookImage()
    #
    line_disp_image.Dxy_axes()
        
        
    ##draw the "ybias_mm" line2
    #(xmin, xmax, ymin, ymax) = line_disp_image.Get_mmBounds()
    ### Draw the effective midpoint in Y (row_bias_mm)
    #line_disp_image.Dline_mm((xmin+20,ybias_mm), (xmax-20,ybias_mm), 'red')


    title='test image'
    cv2.imshow(title, line_disp_image.image)
    cv2.waitKey(0) 
