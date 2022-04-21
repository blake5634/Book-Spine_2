import cv2
import numpy as np
import glob as gb
import copy as cp
import book_parms as bpar
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
#
#  Useful classes 
#
import newfcns as nf

    
def RC2PXY(rcTuple):  ##  OpenCV's confusing Point coords.
    return (rcTuple[1],rcTuple[0])

class bookImage():
    def __init__(self,img, mmPpx):
        self.scale = mmPpx # mm per pixel (or 1 pixel in mm)
        self.image = img
        
        #  set up the size/shape parameters
        sh = self.image.shape
        self.rows = sh[0]
        self.cols = sh[1]
        if len(sh) > 2:
            self.type = 'color'
        else:
            self.type = 'mono'
        self.width_mm =  self.cols*self.scale
        self.height_mm = self.rows*self.scale
        self.ctXmm = self.width_mm/2.0    # mm offset to center of image H
        self.ctYmm = self.height_mm/2.0  # mm offset to center of image V
        
        # validate image type here
        
    def icopy(self):
        return cp.deepcopy(self,{})
    
    def ishape(self):        
        sh = self.image.shape
        
        #print('   Image scale info:')
        #print('   Scale Factor: ', self.scale, ' mm/pixel')
        #print('   rows/cols: ',      sh)
        #print('   height/width(mm)', self.height_mm, self.width_mm )
    
        return sh
        
    def get_px_RC(self,row,col):
        return(self.image[row][col])
    
    def set_px_RC(self, row,col,value):
        if self.type == 'mono':
            l1 =1
        elif self.type == 'color':
            l1 = 3
        else:
            print('set_px_RC:  image type')
            quit()
        if type(value) == type(5):
            l2 = 1
        if type(value) == type((0,0,0)):
            l2 = 3
        assert l1 == l2, 'attempting to set wrong pixel size (RGB vs mono)'
        self.image[row][col] = value
        return
    
    def get_px_XYmm(self,Xmm,Ymm):
        row,col = self.XYmm2RC(Xmm,Ymm)
        return self.image[row,col]
    
    def XYmm2RC(self,Xmm,Ymm):        
        row = int(self.rows/2  - Ymm/self.scale)
        col = int(self.cols/2  + Xmm/self.scale)
        return (row, col)
    
    def RC2XYmm(self,row,col):
        Xmm =   col*self.scale    - self.ctXmm
        Ymm =   -1*row*self.scale + self.ctYmm
        return (Xmm, Ymm)
        
    # create a new bookImage scaled down by factor
    def downvert(self, factor):
        print('downvert: scaling down by factor: ',factor)
        tmp = self.icopy() 
        tmp.scale = self.scale * factor  # mm_per_pixel
        sh = self.image.shape
        img_width  = int(sh[1] / factor)
        img_height = int(sh[0] / factor)
        tmp.rows = img_height
        tmp.cols = img_width
        tmp.image = cv2.resize(tmp.image, (img_width,img_height))
        # set up the size/shape parameters
        tmp.width_mm =  tmp.cols*tmp.scale
        tmp.height_mm = tmp.rows*tmp.scale
        tmp.ctXmm = tmp.width_mm/2.0    # mm offset to center of image H
        tmp.ctYmm = tmp.height_mm/2.0  # mm offset to center of image V
         
        return tmp 

    def blur_mm_rad(radius):
        b = int(radius/self.scale)
        if b%2 == 0:  # radius must be odd # 
            b+=1            
        tmp = cv2.GaussianBlur(self.image, (b,b), 0)
        self.image = tmp
        return 
    
    def Get_mmBounds(self):
        xmin = -self.width_mm/2
        ymin = -self.height_mm/2
        xmax = self.width_mm/2
        ymax = self.height_mm/2
        return (xmin, xmax, ymin, ymax)
            
    def Dline_ld(self,ld,color):  # draw line based on line dict params
        p1 = (ld['xmin'], ld['m0']*ld['xmin'] + ld['b0'] + ld['ybias'])
        p2 = (ld['xmax'], ld['m0']*ld['xmax'] + ld['b0'] + ld['ybias'])
        self.Dline_mm(p1,p2,color)
        
    #  Draw a line/rect in mm coordinates
    #
    #   p1 = (p1X, p1Y) etc
    #      NOTE:   drawing uses "Points()" not [row,col]!!!!!
    def Dline_mm(self, p1xy, p2xy, st_color, width=3):
        p1_px = self.XYmm2RC( p1xy[0], p1xy[1])  # Xmm, Ymm
        p2_px = self.XYmm2RC( p2xy[0], p2xy[1])
        #print('Dline_mm:  Drawing line from {:} to {:} (row,col)'.format(p1_px,p2_px))
        cv2.line(self.image, RC2PXY(p1_px), RC2PXY(p2_px), bpar.colors[st_color], width)
        
        
        
    def Drect_mm(self,  p1, p2, st_color, width=3):
        p1_px = self.XYmm2RC(p1[0], p1[1])
        p2_px = self.XYmm2RC(p2[0], p2[1])
        cv2.rectangle(self.image, RC2PXY(p1_px), RC2PXY(p2_px), bpar.colors[st_color], width)
        
    # draw a square to mark a spot centered on p1
    def Dmark_mm(self, p1mm, side, color):
        #side = square side length in mm 
        # get corners of square
        ps1 = ( p1mm[0] - side/2, p1mm[1] - side/2 ) 
        ps2 = ( p1mm[0] + side/2, p1mm[1] + side/2  )
        self.Drect_mm(ps1,ps2,color)
        
    # draw a square to mark a spot centered on p1
    def Dmark_px(self, p1RC, side, color):
        #side = square side length in pixels 
        # get corners of square
        ps1 = ( int(p1RC[0] - side/2), int(p1RC[1] - side/2) ) 
        ps2 = ( int(p1RC[0] + side/2), int(p1RC[1] + side/2) )
        self.Drect_px(ps1,ps2,color)
        
    def Dline_px(self, p1, p2, st_color, width=3):
        p1r = (p1[1],p1[0])
        p2r = (p2[1],p2[0])
        cv2.line(self.image, p1r, p2r, bpar.colors[st_color], width)

        
    def Drect_px(self, p1, p2, st_color, width=3):
        p1r = RC2PXY(p1)
        p2r = RC2PXY(p2)
        cv2.rectangle(self.image, p1r, p2r, bpar.colors[st_color], width)
         
    def Dxy_axes(self):
         # Draw H and V axes (X,Y axes in mm)    
        (xmin, xmax, ymin, ymax) = self.Get_mmBounds()

        self.Dline_mm((xmin,0), (xmax,0),'white')
        self.Dline_mm((0, ymin), (0, ymax), 'white')

        ## Draw some tick marks
        tick_locs_mm = [] # pix
        tickwidth = 20 # mm
        for xt in range(int(xmax/tickwidth)): # unitless
            xpt = tickwidth*(xt+1)  # mm
            tick_locs_mm.append(xpt)
            tick_locs_mm.append(-xpt)
        ya = 0.0 #mm
        yb = -5.0 #mm
        for x in tick_locs_mm:
            self.Dline_mm((x, ya), (x,yb), 'green')   # draw the tick marks
            
def approx(a,b):
    if abs(a-b) < 0.0001:
        return True
    else:
        return False
        
def printpass(str):
    print('\n                 {:40}         PASS\n'.format(str))
        
        
if __name__=='__main__':
    
    print('\n\n                  Testing bookImage class \n\n')
        
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
        img_orig = cv2.imread(pic_filename, cv2.IMREAD_COLOR)
        tsti = img_orig.copy()  # copy of original for visualization 

        sh = img_orig.shape
        print('Original:   {:} rows, {:} cols.'.format(sh[0],sh[1]))
        
        #Instantiate
        pixels_per_mm = 5.0
        mm_per_pixel = 1.0/pixels_per_mm
        tim1 = bookImage(img_orig, mm_per_pixel)   # mm/pixel
        
        sh2 = tim1.ishape()
        assert sh == sh2, 'shape method FAILS'
        
        printpass('ishape() tests')
        
        ## #  converting tests
        
        ##   TEST mm to row,col  on 4 corners of a squares
        #print('Converted value of XY=( 0.0, 0.0)mm is', tim1.XYmm2RC( 0.0,  0.0))
        #print('Converted value of XY=(50.0,25.0)mm is', tim1.XYmm2RC( 50.0,25.0))
        fs = 'mm to pixel conversion ERROR'
        assert tim1.XYmm2RC( 0.0,  0.0) == (544,810), fs
        assert tim1.XYmm2RC( 50.0,25.0) == (419,1060), fs
        assert tim1.XYmm2RC( 50.0, 0.0) == (544,1060), fs
        assert tim1.XYmm2RC(  0.0,25.0) == (419,810), fs
        
        printpass('XYmm2RC tests')
        
        rtst = 100
        ctst = 250
        
        pxymm = tim1.RC2XYmm(rtst,ctst)
        assert tim1.XYmm2RC(pxymm[0],pxymm[1]) == (rtst,ctst), 'RC->XY->RC FAILS'
        
        printpass('XYmm2RC(RC2XYmm(R,C) test')
        
        ##  TEST row,col to mm
        
        print('Converted value of ', int(tim1.rows/2), int(tim1.cols/2),' is ', tim1.RC2XYmm(tim1.rows/2,tim1.cols/2))
        fs = 'pixel to mm conversion ERROR'
        x = tim1.RC2XYmm(tim1.rows/2,tim1.cols/2)
        assert approx(x[0],0.0), fs
        assert approx(x[1],0.0), fs
    
        
        x = tim1.RC2XYmm(tim1.rows/2+10,tim1.cols/2+20)
        #print(' 10, 20 px more: ', x)
        assert approx(x[0],  4), fs
        assert approx(x[1], -2), fs
        
        printpass('center conversion tests')
        
        ##
        #
        #   Test icopy() vs make new image
        #
        
        tst_icop = tim1.icopy()
        tst_newbI = bookImage(tim1.image, tim1.scale)
        
        #assert tst_icop.image == tst_newbI.image,  'image mismatch'
        assert tst_icop.scale     == tst_newbI.scale,     'scale mismatch'
        assert tst_icop.height_mm == tst_newbI.height_mm, 'height_mm mismatch'
        assert tst_icop.width_mm  == tst_newbI.width_mm,  'width_mm mismatch'
        assert tst_icop.rows      == tst_newbI.rows,      'rows mismatch'
        assert tst_icop.cols      == tst_newbI.cols,      'cols mismatch'
        assert tst_icop.ctXmm     == tst_newbI.ctXmm,     'ctXmm mismatch'
        assert tst_icop.ctYmm     == tst_newbI.ctYmm,     'ctYmm mismatch'

        printpass('copy vs. newinstance')
        
        
        #
        #   test drawing
        #
        
        # Place marks in right places:
        
        tim1.Dmark_mm((  0.0, 0.0), 2, 'red')
        tim1.Dmark_mm(( 10.0, 0.0), 2, 'red')
        tim1.Dmark_mm(( 20.0,10.0), 2, 'red')
        tim1.Dmark_mm(( -10.0,-50.0), 2, 'red')
        
        # test rectangle drawing
        #  (should be predominantly horizontal)
        tim1.Drect_mm( (-80,-20), (-10,-10), 'green')
        
        # pixel line from one corner (almost) to the other
        tim1.Dline_px( (10,10), (1079,1610),'red')
        
        # draw some test lines in px:
        
        for i in range(10):
            r = 50*i
            midpoint = int(tim1.cols/2)
            p1 = (r,midpoint)
            p2 = (r,midpoint+500)
            tim1.Dline_px(p1,p2,'yellow',width=2)
        
        for i in range(10):
            ymm = 100 - 50*tim1.scale*i
            midpoint = 0.0
            xmm = midpoint
            p1 = (xmm,              ymm)
            p2 = (xmm+500*tim1.scale, ymm)
            tim1.Dline_mm(p1, p2,'white')
        
        tim1.Dline_px( (544,810), (544-250,810+125), 'white')
        
        # pixel line from origin to +x and +y values
        tim1.Dline_mm( (0,0), (50,25), 'green')
            
         
        title='test image'
        cv2.imshow(title, tim1.image)
        cv2.waitKey(5000)
        
        #######################################################################33   Scale Down
        #
        #  Now try similar tests with scaled image
        #
        #tim2 = bookImage(cv2.imread(pic_filename, cv2.IMREAD_COLOR), mm_per_pixel).downvert(4.0)
        print('\n\n----------------------------------')
        print('original image;')
        tim1.ishape()
        tim2 = tim1.icopy()
        print('generating scaled test image: ')
        tim2 = tim1.downvert(4.0)        
        tim2.ishape()

        #tim2.Dxy_axes()
        tim2.Dline_px((100,100),(200,400),'green') 
        tim2.Dmark_px((200,200),14,'yellow')
        tim2.Dmark_px((200,300),14,'blue')
        
        
        # Plot a line in mm on scaled image
        th = 145
        xintercept = 80
        llength = bpar.book_edge_line_length_mm
        bias = -20
        ld2 = nf.Get_line_params(th, xintercept, llength, bias)
        tim2.Dline_ld(ld2,'white')
        
        ###  draw mm coordinate system
        tim2.Dxy_axes()
        
        cv2.imshow('Scaled Down Test',tim2.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
        
