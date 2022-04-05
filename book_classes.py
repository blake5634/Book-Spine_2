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

class bookImage():
    def __init__(self,img, mmPpx):
        self.scale = mmPpx # mm per pixel
        self.image = img
        self.ishape()
        # validate image type here
        
        
    def get_px_RC(self,row,col):
        return(self.image[row][col])
    
    def get_px_XYmm(self,X,Y):
        row,col = self.XYmm2RC(X,Y)
        return self.image[row,col]
    
    def XYmm2RC(self,X,Y):        
        row = int((self.ctY-Y)/self.scale)
        col = int((self.ctX+X)/self.scale)
        return row, col
    
    def RC2XYmm(self,row,col):
        X = col*self.scale+self.ctX
        Y = -(row*self.scale-self.ctY)
        return X, Y
    
    def ishape(self):        
        sh = self.image.shape
        self.rows = sh[0]
        self.cols = sh[1]
        self.ctX = self.cols*self.scale/2  # mm offset to center of image H
        self.ctY = self.rows*self.scale/2  # mm offset to center of image V
        return sh
        
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
    #  if image scale is different from "scale" then use param
    #   p1 = (p1X, p1Y) etc
    def DLine_mm(self, p1, p2, st_color, width=3):
        p1_px = self.XYmm2RC( p1[0], p1[1])
        p2_px = self.XYmm2RC( p2[0], p2[1])
        print('Drawing line from ',p1_px, ' to ', p2_px)
        cv2.line(self.image, p1_px, p2_px, bpar.colors[st_color], width)
        
    def DRect_mm(self,  p1, p2, st_color, width=3):
        p1_px = self.XYmm2RC( p1[1], p1[0])
        p2_px = self.XYmm2RC( p2[1], p2[0])
        cv2.rectangle(self.image, p1_px, p2_px, bpar.colors[st_color], width)
        
    def DLine_px(self, p1, p2, st_color, width=3):
        cv2.line(self.image, p1, p2, bpar.colors[st_color], width)

        
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
        ish = img_orig.shape
        tsti = img_orig.copy()  # copy of original for visualization 

        sh = img_orig.shape
        print('Original:   {:} rows, {:} cols.'.format(sh[0],sh[1]))
        
        #Instantiate
        tim1 = bookImage(img_orig, 5.0)   # 5mm/pixel
        
        sh2 = tim1.ishape()
        assert sh == sh2, 'shape method FAILS'
        
        #
        #   test drawing
        #
        
        # line
        #tim1.DLine_mm( (0,0), (25,25), 'green')
        tim1.DLine_px( (0,0), (1620,1089),'red')
        #tim1.DRect_mm( (0,0), (25,25), 'blue')
        
        title='test image'
        cv2.imshow(title, tim1.image)
        cv2.waitKey(5000)
            
        
        
