import cv2

scale = 2  # downsample image by this much
d2r = 3.141592638*2/360.0

#   ~5.0 pix/mm  (measured from original unscaled target img)

pix2mm = 0.2 * float(scale)   # convert pix * pix2mm = xx mm
                                    # measure off unscaled target image
mm2pix = float(1.0)/pix2mm 


image_dir = 'tiny/'
tmp_img_path = 'tmp_images/'
################################################
#
#    Image input stage and color VQ params
#

#  following smoothing windows are scaled from here by /scale
#     values below reflect a nominal image width of 1670

deriv_win_size =     int(1.0*mm2pix)      # 1mm width
smooth_size    = -1* int(10*mm2pix)    # <0:   do not smooth

blur_flag = False
blur_rad_mm       =     int(4.0)    # image must determine # pixels

KM_Clusters = 20  # number of K-means clusters for color
 

######################################################
#
#    LINE SCORE PARAMS
#
Line_Score_Thresh = 0.300 # score units (lower is better) 

max_gap_mm = 3  #mm   gap btwn lines defining new cluster (slant.py)

# how long is average book?
book_edge_line_length_mm = 120 # mm  line length checked for edges
slice_width  = 5 # mm   width examined for color dominance on either side of line
row_bias_mm = -20  # mm   (shift line down from Y=0 line)         
min_line_vals = 25  # min number of points above and below "0" 

#  method 1 == Euclidean RGB distance
#  method 2 == 350 if labels different, 0 otherwise
Color_Dist_Method = 1

###################
#
#  Rank based selection of line candidates
topNbyscore = 50   # show only the N best lines


##########
#
#   Line cluster params
line_VQ_Nclusters = 7

###########
#
#  clusterneighborhood search

KMneighborDX =  5 #mm
KMneighborDth = 12 # deg


##################################################
#
#  Cluster cleanup and bookfinding
#

esize = 8   # erode px
dsize = 10  # dilate px


###################################################
#
#    Cluster/Blob selection parameters
#

area_min = 2000
elong_min = 4
elong_max =  100
noise_area_threshold = 100

corner_dist_max_px = 10
enough_corners = 2




font = cv2.FONT_HERSHEY_SIMPLEX
colors = {'black':(0,0,0), 'white':(255,255,255), 'blue':(255,0,0), 'green':(0,255,0), 'red':(0,0,255),'yellow':(0,255,255),
          'acqua':(255,255,0),'fuchsia':(255,0,255),'maroon':(0,0,128),'navy':(128,0,0),'olive':(0,128,128)}
'''
    Black: (0, 0, 0)
    White: (255, 255, 255)
    Red: (255, 0, 0)
    Green: (0, 255, 0)
    Blue: (0, 0, 255)
    Aqua: (0, 255, 255)
    Fuchsia: (255, 0, 255)
    Maroon: (128, 0, 0)
    Navy: (0, 0, 128)
    Olive: (128, 128, 0)
    Purple: (128, 0, 128)
    Teal: (0, 128, 128)
    Yellow: (255, 255, 0)
'''
