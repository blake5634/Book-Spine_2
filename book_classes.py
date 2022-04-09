import cv2
import numpy as np
import glob as gb
import book_parms as bpar
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
#
#  Useful classes 
#


    
def RC2PXY(rcTuple):  ##  OpenCV's confusing Point coords.
    return (rcTuple[1],rcTuple[0])

class bookImage():
    def __init__(self,img, mmPpx):
        self.scale = mmPpx # mm per pixel
        self.image = img
        self.ishape()
        # validate image type here
    
    def ishape(self):        
        sh = self.image.shape
        self.rows = sh[0]
        self.cols = sh[1]
        self.width_mm =  self.cols*self.scale
        self.height_mm = self.rows*self.scale
        self.ctXmm = self.width_mm/2.0    # mm offset to center of image H
        self.ctYmm = self.height_mm/2.0  # mm offset to center of image V
        
        print('   Image scale info:')
        print('   rows/cols: ',      sh)
        print('   height/width(mm)', self.height_mm, self.width_mm )
    
        return sh
        
    def get_px_RC(self,row,col):
        return(self.image[row][col])
    
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
        tmp = self.copy()
        sh = tmp.shape()
        img_width  = int(sh[1] / factor)
        img_height = int(sh[0] / factor)
        tmp.image = cv2.resize(tmp.image, (img_width, img_height))
        tmp.scale = self.scale * factor
        return tmp
            
    #  Draw a line/rect in mm coordinates
    #
    #   p1 = (p1X, p1Y) etc
    #      NOTE:   drawing uses "Points()" not [row,col]!!!!!
    def Dline_mm(self, p1xy, p2xy, st_color, width=3):
        print('mmDrawing line from ',p1xy, 'mm  to ', p2xy, 'mm')
        p1_px = self.XYmm2RC( p1xy[0], p1xy[1])  # Xmm, Ymm
        p2_px = self.XYmm2RC( p2xy[0], p2xy[1])
        #point1 = 
        print('mmDrawing line from ',p1_px, ' to ', p2_px)
        cv2.line(self.image, RC2PXY(p1_px), RC2PXY(p2_px), bpar.colors[st_color], width)
        
    def DRect_mm(self,  p1, p2, st_color, width=3):
        p1_px = self.XYmm2RC( p1[1], p1[0])
        p2_px = self.XYmm2RC( p2[1], p2[0])
        cv2.rectangle(self.image, RC2PXY(p1_px), RC2PXY(p2_px), bpar.colors[st_color], width)
        
    def Dline_px(self, p1, p2, st_color, width=3):
        p1r = (p1[1],p1[0])
        p2r = (p2[1],p2[0])
        cv2.line(self.image, p1r, p2r, bpar.colors[st_color], width)

        
    def DRect_px(self, p1, p2, st_color, width=3):
        p1r = RC2PXY(p1)
        p2r = RC2PXY(p2)
        cv2.rectangle(self.image, p1r, p2r, bpar.colors[st_color], width)
        
    def copy(self):
        tmp = self() 
        tmp.image = self.image
        tmp.scale = self.scale
        tmp.ishape()
        return tmp

def approx(a,b):
    if abs(a-b) < 0.0001:
        return True
    else:
        return False
        
        
        
        
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
        tim1 = bookImage(img_orig, 1/pixels_per_mm)   # mm/pixel
        
        sh2 = tim1.ishape()
        assert sh == sh2, 'shape method FAILS'
        
        
        ## #  converting tests
        
        ##   TEST mm to row,col
        #print('Converted value of XY=( 0.0, 0.0)mm is', tim1.XYmm2RC( 0.0,  0.0))
        #print('Converted value of XY=(50.0,25.0)mm is', tim1.XYmm2RC( 50.0,25.0))
        fs = 'mm to pixel conversion ERROR'
        assert tim1.XYmm2RC( 0.0,  0.0) == (544,810), fs
        assert tim1.XYmm2RC( 50.0,25.0) == (419,1060), fs
        
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
        
        #
        #   test drawing
        #
        
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
            
        
        if False:# line
            rS2 = 25/2.0
            tim1.DRect_mm( (-rS2,-rS2), (rS2,rS2), 'blue')
            
            
            #1089 rows, 1620 cols.
            rS2px = 3
            rowctr = int(1089/2)
            assert rowctr == int(sh[0]/2)
            colctr = int(1620/2)
            assert colctr == int(sh[1]/2)
            
            tim1.DRect_px((rowctr-rS2px, colctr-rS2px),(rowctr+rS2px, colctr+rS2px),'white')
        
        
        title='test image'
        cv2.imshow(title, tim1.image)
        cv2.waitKey(5000)
            
        
        
