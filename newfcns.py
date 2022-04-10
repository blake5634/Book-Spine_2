import cv2
import numpy as np
import glob as gb
import book_parms as bpar
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
#
#  new functions by BH
#
##
d2r = 2*np.pi/360.0

##s
# Convert XY rel LL corner to row, col
#
##  from   XY referenced to center of img:
#
#                       |Y
#                       |
#                       |
#      -----------------------------------
#         X             |\
#                       |  (img_width_cols/2,img_height_rows/2)
#                       |
#                       |
#
#    to classic image proc coordinates:
#      0/0------- col ----->
##      |
#       row
#       |
#       |
#       V
##
def Get_pix_byXY(img,X,Y,iscale=bpar.scale):
    #print('Get: {} {}'.format(X,Y))
    row,col = XY2RC(img,X,Y,iscale==iscale)
    #if col > 500:
        #print('  Get_pix_byXY() X:{} Y:{} r:{} c:{}'.format(X,Y,row,col))
    return(img[row,col])

def Get_pix_byRC(img,row,col):
    #print('        get pix: R: {} C: {}'.format(row,col))
    return(img[row,col])

#
# convert image ctr XY(mm) to X,Y (open CV points)
#
def XY2iXiY(img,X,Y,iscale=bpar.scale):
    if iscale != bpar.scale:   # bpar.mm2pix has scale in it so..
        f = float(bpar.scale)/float(iscale)
        X *= f
        Y *= f
    row = int( -Y*bpar.mm2pix + int(img.shape[0]/2) )
    col = int(  X*bpar.mm2pix + int(img.shape[1]/2) )
    iX = col
    iY = row
    #return iX, iY
##
## convert image ctr XY(mm) to Row, Col 
##
#def XY2RC(img,X,Y,iscale=bpar.scale):
    #if iscale != bpar.scale:
        #f = float(bpar.scale)/float(iscale)
        #X *= f
        #Y *= f
    #row = int( -Y*bpar.mm2pix + int(img.shape[0]/2) )
    #col = int(  X*bpar.mm2pix + int(img.shape[1]/2) )
    #return row,col

#
#  Get image bounds in mm  
#
def Get_mmBounds(img,iscale=bpar.scale):
    sh = np.shape(img)  # get rows & cols
    xmin = -1* (sh[1]*bpar.pix2mm)  # bpar.pix2mm factor includes bpar.scale
    xmax = -1*xmin
    ymin = -1* (sh[0]*bpar.pix2mm)
    ymax = -1*ymin
    if iscale != bpar.scale:
        f = float(bpar.scale)/float(bpar.scale)
        xmin *= f
        xmax *= f
        ymin *= f
        ymax *= f
    return (xmin, xmax, ymin, ymax)
#
#  Draw a line/rect in mm coordinates
#
#  if image scale is different from "scale" then use param
#
def DLine_mm(img, p1, p2, st_color, width=3,iscale=bpar.scale):
    p1_pix = XY2iXiY(img, p1[0],p1[1], iscale=iscale)
    p2_pix = XY2iXiY(img, p2[0],p2[1], iscale=iscale)    # allows for change of scale 
    cv2.line(img, p1_pix, p2_pix, bpar.colors[st_color], width)
    
def DRect_mm(img,  p1, p2, st_color, width=3,iscale=bpar.scale):
    p1_pix = XY2iXiY(img, p1[0],p1[1], iscale=iscale)
    p2_pix = XY2iXiY(img, p2[0],p2[1], iscale=iscale)
    cv2.rectangle(img, p1_pix, p2_pix, bpar.colors[st_color], width)
 
#
#  Return standard image size references
#    img = unscaled image
#    scale = int scale factor to be used
def Get_sizes(img, scale):
    ish = np.shape(img)
    simg_width_cols = int(ish[1]/scale)
    simg_height_rows =  int(ish[0]/scale)
    return ish, (simg_height_rows, simg_width_cols)

def Get_line_params(th, xintercept, llength, bias, w=None):
    '''
    th = angle in degrees
    xintercept = in mm
    llength = length of line (mm)
    ybias = location of "y=0" line (vertical shift)
    w = width of analysis window (perp to line)
    
    returns dictionary of line info
    '''
    d = {}
    d['th'] = th
    d['xintercept'] = xintercept
    d['m0'] = np.tan(th*d2r) # slope (mm/mm)
    d['b0'] = -d['m0']*xintercept  # mm
    d['ybias'] = bias
    
    
    # compute xmin and xmax
    x_line_len_mm = abs((llength)*np.cos(th*d2r)) # mm
    if x_line_len_mm < 0.5:
        x_line_len_mm = 2.0

    d['xmin'] = xintercept - x_line_len_mm/2.0
    d['xmax'] = xintercept + x_line_len_mm/2.0

    #  window upper bound (window width (mm) is perp to line but we need to go straight up)
    #  rV = distance to upper bound (vertical) mm
    if w:
        d['w'] = w
        d['rV'] = abs(w/np.sin(th*bpar.d2r-np.pi/2)) # mm  (a delta so there's no origin offset)
        tmp = int(d['rV'] * bpar.mm2pix)     # pixels
        if tmp < 2:  # prevent tiny windows
            tmp = 2
        d['rVp'] = tmp
         
    #d['ybias_mm'] = -d['b0']/d['m0']  # y intercept
    assert (d['xintercept'] > d['xmin'] and d['xintercept'] <= d['xmax']), 'bad x-value: '+str(d['xintercept'])

    return d






#
#
#  Get edge score of a line through image
#
#   y = mx+b  (y=row, x=col)
#
#   1) strike a vertical at each x value
#   2) determine distance up/down to scan to reach window width w (rV, rVp)
#   3) append all VQ labels along that vertical
#   4) find dominant color label on either side of line- look for uniformity
#
#   NEW:   All coordinates and radii etc are in mm 
def Get_line_score(img, w, ld, cdist):
    '''
    img = bookImage() class
    w   = width of line analysis window (90deg from line) (mm)
    ld = line parameter dict:
        ld['xintercept'] = where line crosses vertical centerline of image (X=0) (mm)
        ld['th'] = angle in deg relative to 03:00 on clock
        ld['llen'] = length of line segment (mm)
        ld['bias'] = vertical shift of the line center (mm)
    cdist = matrix of color distances (Euclid) btwn VQ centers
    '''
    #print('\n\nLine Score:   x: {}(mm) , th: {}(deg)'.format( ld['xintercept'], th))
    #print(' ---   image shape: {}'.format(np.shape(img)))
    #print('Image sample: {}'.format(img[10,10]))
    img_height_rows = img.rows # height/rows
    img_width_cols = img.cols 
    
    m0 = ld['m0'] # slope (mm/mm)
    b0 = ld['b0']  # mm
    rV = ld['rV']
    rVp = ld['rVp'] 
     
     
    #print('xmin/max2: {:4.2f}mm {:4.2f}mm'.format(xmin2,xmax2))
    # cols,  rows = XY2iXiY()
    
    print('xmin/max mm: ', ld['xmin'], ld['xmax'])
    # xmm --> col#
    dummy, xmin_px = img.XYmm2RC(ld['xmin'],0)  # pix  X range for test line
    dummy, xmax_px = img.XYmm2RC(ld['xmax'],0)  #  --> pixels in opencv coords.
    
    xmin_px = max(0,xmin_px)                  #  just in case
    xmax_px = min(img_width_cols,xmax_px)
    
    
    
    # range of xvals for each line
    xrange_px = range(xmin_px, xmax_px-1, 1)  # pix cols
    
    print('x range: {} -- {}'.format(xmin_px, xmax_px)) 
    #study pixels above and below line at all columns in range   
    vals_abv = []  # values above the line (for all x values)
    vals_bel = []  #        below 
    ncolsLine = 0
    nvals_app = 0 
    for col in xrange_px:  # for each x-value, go vertically above & below line.
        ncolsLine += 1
        x,y = img.RC2XYmm(0, col) # convert back to mm(!)
        ymm = m0*x+b0 + ld['ybias']    # line eqn in mm
        row, dummy = img.XYmm2RC(0,ymm)    # pix
        #print ('X/col:{} Ymm:{:4.2f} row:{}'.format(col,ymm,row))
        print('checking row and col: ', row, col)
        if not ((row > img_height_rows-1 or row < 0) or (col > img_width_cols-1 or col < 0)): # line not outside image?
            # above the line
            print('looking at rows: ', row-rVp, row)
            for row1 in range(row,row-rVp,-1): # higher rows #s are "lower"
                if  row1 > 0:
                    print('            Looking at (above): {}, {}'.format(row1,col))
                    #vals_abv.append(Get_pix_byRC(img,row1,col)) # accum. labels in zone above
                    vals_abv.append(img.image[row1,col]) # accum. labels in zone above
                    nvals_app+=1
            # below the line
            for row1 in range(row, row+rVp,1):
                if row1 < img_height_rows:  # 
                    #print('            Looking at (below): {}, {}'.format(row1,col))
                    #vals_bel.append(Get_pix_byRC(img,row1,col))
                    vals_bel.append(img.image[row1,col])
                    nvals_app+=1
                
    
    #
    #  If we got enough pixels in our window for meaningful result:
    #
    if len(vals_abv) > bpar.min_line_vals and len(vals_bel) > bpar.min_line_vals: 
        #print('shape vals: {}'.format(np.shape(vals_abv)))
        #print('sample: vals: ', vals_abv[0:10])
        labs_abv, cnts_abv = np.unique(vals_abv, return_counts=True)
        labs_bel, cnts_bel = np.unique(vals_bel, return_counts=True)
        #print('shape: labels_abv: {}, counts_abv: {}   Data: '.format(np.shape(labs_abv),np.shape(cnts_abv)))
        #print(labs_abv, cnts_abv)
        #print('labels above, below: (1st 100 samples)')
        #print(vals_abv[0:100], vals_bel[0:100])
        
        #print('GLP: labels above: ', labs_abv)
        #print('GLP: counts above: ', cnts_abv)
        #print('GLP: labels below: ', labs_bel)
        #print('GLP: counts below: ', cnts_bel)
        
        dom_lab_abv = labs_abv[np.argmax(cnts_abv)]   # which is most common label above?
        dom_lab_bel = labs_bel[np.argmax(cnts_bel)]
        dom_abv = np.max(cnts_abv)/np.sum(cnts_abv)  # how predominant? (0-1)
        dom_bel = np.max(cnts_bel)/np.sum(cnts_bel)  # how predominant? (0-1)
        
        print('\n\nGLP: Dominant labels: above: {:5} below: {:5}'.format(dom_lab_abv,dom_lab_bel))
        print('GLP: Dominance:       above: {:5.3f} below: {:5.3f}\n\n'.format(dom_abv,dom_bel))
        # is the dominant color above line == dom color below?
        Method = bpar.Color_Dist_Method
        if Method == 1:
            color_distance = 0.01* cdist[dom_lab_abv, dom_lab_bel] + 0.1  # keep non zero
        elif Method == 2:
            if dom_lab_abv != dom_lab_bel:
                color_distance = 150    # to match typical color distances
            else:
                color_distance = 0.0
        else:
            print('GLP: Illegal color Distance Method (1 or 2)')
            quit()
        diff_score = color_distance/(dom_abv*dom_bel)  # weighted difference (smaller is better!)
    else:
        #x = input('\n\n\n[enter] to continue (0)')
        return 99999999999  # a really really bad score
    
    #x = input('\n\n\n[enter] to continue ({:5.2f})'.format(diff_score))
    return diff_score
                    
                
            
#
#  Check along top for typical bacgkround pixels
#

##########################
#
#   Look across top of image for label of "black"
#
def Check_background(lab_image, outfile=False):
    bls = []
    wheight = int(lab_image.shape[0]/8)
    for col in range(lab_image.shape[1]):
        for r in range(wheight):
            bls.append(lab_image[r+5,col])
    labs, cnt = np.unique(bls, return_counts=True)
    max = np.max(cnt)
    for l,c in zip(labs,cnt):
        if c == max:
            lmax = l
            break
    print('Most common label near top: {} ({}%)'.format(lmax, 100*max/np.sum(cnt)))
    if outfile:
        f = open('metadata.txt','w')
        print('dark label, {}'.format(lmax), file=f)
        f.close()
    return lmax


#
#  Apply criteria to gaps
#
def Gap_filter(gaps,img, tmin=20, blacklabel=6):
    img_height=img.shape[0]
    img_width=img.shape[1]
    # tmin:        min width pixels
    # blacklabel:  int label of background black

    halfway = int(img_height/2)
    candidates = []
    for g in gaps:
        #
        #  exclude
        #
        # 1) narrow gaps
        width = abs(g[0]-g[1])
        if width < tmin:
            print('found a very narrow gap')
            continue
        # 2) gaps that match background
        values = []
        for c1 in range(width):
            col = g[0]+c1-1
            for r in range(halfway):
                row = halfway + r -1 
                if col < img_width:
                    values.append(img[row,col])
        (val,cnts) = np.unique(values, return_counts=True)
        if val[np.argmax(cnts)] == blacklabel:  # background
            print('found a black gap')
            continue
        else: # we didn't exclude this gap
            candidates.append(g)
    return candidates
#
#
#
def Gen_gaplist(cross): 
    cm1=0
    gaps = []
    for c in cross:
        if c < 0:
            c = c*-1
        gaps.append([cm1,c])
        cm1 = c
    return gaps

#
#  neg and pos zero crossings
#

def Find_crossings(yvals):
    ym1 = yvals[0]
    c = []
    for i,y in enumerate(yvals):
        if y<0 and ym1>=0:
            c.append(-i)   # - == neg crossing
        if y>0 and ym1 <= 0:
            c.append(i)    #   positive crossing
        ym1 = y
    return c
            

def Est_derivative(yvals, w):
    if w < 0:
        return np.gradient(line,2)
    if w > len(yvals)/2:
        print(' derivative window {} is too big for {} values.'.format(w,len(yvals)))
        quit()
    else:
        ym1 = yvals[0]
        dydn = []
        dn = 1 # for now
        for y in yvals:
            dy = y-ym1
            dydn.append(dy/dn)
            ym1 = y
        if w>1:
            dydn = smooth(dydn, window_len=w, window='hanning')        
        return dydn
 
    
def Find_edges_line(line):
    
    #grad = np.gradient(line,2)
    
    grad = Est_derivative(line, 3)
        
    thresh = 0.1
    edges = []
    for i,gv in enumerate(grad):
        if abs(gv) > thresh:
            edges.append(i)

    # get "edge of edges"
    e2 = []
    ep = edges[0]
    for e in edges:
        if e-ep > 1:    # leading edge of each gradient peak
            e2.append(e)
        ep = e
    edges = e2
    
    
        
    return edges

#
#   select a horizontal strip
#    return a series of labels
# 

def Trancept_labeled(lab_img, yval):
    img_height=img.shape[0]
    img_width=img.shape[1]    
    r = int(yval)
    result = []
    for c in range(img_width):
        lab_pix = lab_img[r,c]
        result.append(lab_pix)
    return result
#
#   Same as Trancept but return most common 
#     label in a vertical bar of width bw
#
#  cluster mean = avg value of pixel labels

def Trancept_bar_labeled(lab_img, yval,bw):    
    img_height=lab_img.shape[0]
    img_width=lab_img.shape[1]
    y_val = int(yval) # y=row, x=col
    result = []
    offset = int(bw/2)
    vv_array = []
    for x in range(img_width):
        vertvals = []
        for i in range(bw):
            y = y_val-offset + i
            if y < img_height:
                vertvals.append(lab_img[y,x])
        if len(vertvals) > 1:
            (val,cnts) = np.unique(vertvals,return_counts=True)
            result.append(val[np.argmax(cnts)]) # return most common label in the vert bar
        else:
            result.append(-2)
        vv_array.append(vertvals)
    return result, vv_array


#
#  generate an image illustrating the KM cluster center colors
#
def Gen_cluster_colors(centers):
    FILLED = -1
    rowh = 70
    img_height_rows = rowh * bpar.KM_Clusters
    img_width_cols = 300
    img = np.zeros((img_height_rows,img_width_cols,3), np.uint8)
    y=0
    x=0
    for i in range(len(centers)):
        col = tuple([int(x) for x in centers[i]])
        print('Color label {}: '.format(i),col) 
        cv2.rectangle(img, (x,y), (img_width_cols,y+rowh), col, FILLED)
        cv2.putText(img, 'cluster: {}'.format(i), (50, y+50), bpar.font, 1, (255, 255, 0), 2, cv2.LINE_AA)
        y=y + rowh
    return img
#   
#  Cluster colors by K-means
#
def KM(img,N):
    img_height=img.shape[0]
    img_width=img.shape[1]
    pixels = np.float32(img.reshape(-1, 3))
    # set up some params
    n_colors = N
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .05)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # perform the clustering
    _, labels, (centers) = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    
    
    # dominant color is the palette color which occurs most frequently on the quantized image:
    dominant = centers[np.argmax(counts)]  # most common color label found
    
    labeled_image = labels.reshape(img.shape[:-1])
  
    #print('shape labels: {}'.format(labels.shape))
    #labeled_image = labels.reshape(img.shape[:-1])
    #print('shape labels: {}'.format(labels.shape))
    #print('shape labeled_image: {}'.format(labeled_image.shape))
    
    #print('i (n pix)    Pallette')
    #for i in range(N):
        #print('{}  ({}) '.format(i,counts[i]), centers[i])
     
    # compute a distance matrix between the cluster centers (color similarity)
    dist = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            dist[i,j] = cv2.norm(centers[i]-centers[j])
            
    # from float back to 8bit
    centers = np.uint8(centers)
    labels = labels.flatten()  # need this vis next line?    
    newimg = centers[labels.flatten()]
    
    #reshape
    newimg = newimg.reshape(img.shape)
    
    return [newimg, labeled_image, centers, dist]


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    #assert x.ndim != 1, "smooth only accepts 1 dimension arrays."

    #print ('input data:', len(x))
    #print (x)
    assert (len(x) > window_len),"Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if window_len%2 == 0:
        print(' smoothing window length must be ODD')
        quit()
        
    assert (window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']), "Window must be: 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    #return y to original length
    extra = len(y)-len(x)
    endchop = int(extra/2)
    print('extra: {}, endchop: {}'.format(extra, endchop))
    z = y[endchop:-endchop]
    print('Orig len: {}  New len: {}'.format(len(x), len(z)))
    return z

