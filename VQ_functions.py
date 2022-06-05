import cv2
import numpy as np
import glob as gb
import book_parms as bpar
import book_classes as bc
import numpy as np
 
import pickle     # for storing pre-computed K-means clusters
import sys as sys
import os as os
#
#  new functions by BH
#
##
d2r = 2*np.pi/360.0 
#
#  Check along top for typical bacgkround pixels
#

##########################
#
#   Look across top of image for label of "background color"
#
def Check_background(lab_image, top_bottom, outfile=False):
    bls = []
    if top_bottom not in ['BG_TOP', 'BG_BOTTOM']:
        print('Unknown background region: ', top_bottom, '     ... quitting.')
        quit()
    # height of background estimation window in rows
    wheight = int(lab_image.shape[0]/12)
    if top_bottom == 'BG_TOP':
        rows_to_check = range(wheight)
    elif top_bottom == 'BG_BOTTOM':
        rows_to_check = range(lab_image.shape[0]-wheight, lab_image.shape[0])
    for col in range(lab_image.shape[1]):
        for r in rows_to_check:
            bls.append(lab_image[r,col])
    labs, cnt = np.unique(bls, return_counts=True)
    print('TEST:  ')
    print (labs)
    print (cnt)
    print('------')
    max = np.max(cnt)
    for l,c in zip(labs,cnt):
        if c == max: # TODO:   argmax!!!
            lmax = l
            break
    print('Most common label {}: {} ({:5.2f}%)'.format(top_bottom, lmax, 100*max/np.sum(cnt)))
    if outfile:
        f = open('metadata.txt','a')
        print('{}, {}'.format(top_bottom, lmax), file=f)
        f.close()
    return lmax
 
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
    N = len(centers)
    #  we're going to set up a grid of color patches 
    gr = int((N+1)/2)
    gc = 2             # grid cols
    rowh = 70
    col_width = 300 
    img = np.zeros((gr*rowh,gc*col_width,3), np.uint8)
    ctr = 0
    for row in range(gr):
        for col in range(gc):
            if ctr < len(centers):
                color = tuple([int(x) for x in centers[ctr]])
                print('Color label {}: '.format(ctr),color) 
                x = col * col_width
                y = row * rowh
                cv2.rectangle(img, (x,y), (x+col_width,y+rowh), color, FILLED)
                cv2.putText(img, 'cluster: {}'.format(ctr), (x+50, y+50), bpar.font, 1, (255, 255, 0), 2, cv2.LINE_AA)
                ctr += 1
    return img
#   
#  Cluster colors by K-means
#
def KM(img,N):
    img_height=img.shape[0]
    img_width=img.shape[1]
    pixels = np.float32(img.reshape(-1, 3))
    # set up some params
    n_centers = N
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .05)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # perform the clustering
    _, labels, (centers)  = cv2.kmeans(pixels, n_centers, None, criteria, 10, flags)
    _, counts             = np.unique(labels, return_counts=True)
    
    
    # dominant color is the palette color which occurs most frequently on the quantized image:
    dominant = centers[np.argmax(counts)]  # most common color label found
    
    codeword_image = labels.reshape(img.shape[:-1])
  
    #print('shape labels: {}'.format(labels.shape))
    #codeword_image = labels.reshape(img.shape[:-1])
    #print('shape labels: {}'.format(labels.shape))
    #print('shape codeword_image: {}'.format(codeword_image.shape))
    
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
    
    #   newimg:   an image where each pixel gets the color of its VQ codeword
    #   codeword_image:  a gray image where each pixel has the label number
    #   centers:   an array of the codeword centers (BGR points)
    #   dist:   a 2x2 matrix of distances between codeword centers    
    return [newimg, codeword_image, centers, dist]



#   
#  Cluster scored lines by K-means
#     according to their XY center
#
def KM_ld(sc_lines,n_centers):
    '''
    sc_lines  =  list of [s, ld ] lists
    '''
    xyth_centers = np.zeros((len(sc_lines),3),dtype=np.float32)
    for i,scl in enumerate(sc_lines):
        ld = scl[1]
        x = ld['xintercept']
        y = ld['ybias']
        xyth_centers[i][0] = np.float32(x)
        xyth_centers[i][1] = np.float32(y)
        xyth_centers[i][2] = np.float32(ld['th'])
    # set up some params
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .05)
    flags = cv2.KMEANS_RANDOM_CENTERS
    number_attempts = 10
    # perform the clustering
    _, labels, (centers)  = cv2.kmeans(xyth_centers, n_centers, None, criteria, number_attempts, flags)
    _, counts             = np.unique(labels, return_counts=True)
      
    return centers,counts



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
                #quit()
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
                img0, tmp, ctrs, color_dist = KM(image, N)   
                pick_payload = [img0, tmp, ctrs, color_dist]
                pickle.dump(pick_payload, pf, protocol=pprotocol)
                pf.close()
    
    return pick_payload
 
