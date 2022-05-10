#$!/usr/bin/python
#
#    Pipeline for finding books featuring result caching
#
  
import numpy as np
import cv2
    
    
#  book finding functions from this project
import book_classes as bc 
import book_parms as bpar 
import newfcns as nf

import time
import sys
import os as os
import pickle     # for storing data btwn steps

#import glob as gb
import matplotlib.pyplot as plt


# where the pickles will live
pickle_dir = 'VQ_pickle/'

def step00():
    i = 0
    print('Starting Step ', i)
    nm = 'step{:02d}'.format(i)
    if ifpick(nm):
        pl = readpick(nm)
    else:
        time.sleep(4)
        writepick(nm, range(50))
    
def step01():
    i = 1
    print('Starting Step ', i)
    nm = 'step{:02d}'.format(i)
    if ifpick(nm):
        pl = readpick(nm)
    else:
        time.sleep(4)
        writepick(nm, range(50))
    
def step02():
    i = 2
    print('Starting Step ', i)
    nm = 'step{:02d}'.format(i)
    if ifpick(nm):
        pl = readpick(nm)
    else:
        time.sleep(4)
        pl = range(50)
        writepick(nm, pl)
        
        
    print('Final result: ', pl) 
    



def ifpick(name):
    #
    #  check for file
    #
    pfname = pickle_dir + name + '_pickle.p'
    if not os.path.isdir(pickle_dir):  # if this doesn't exist, create it.
        print('Creating a new pickle directory: ./'+pickle_dir)
        os.mkdir(pickle_dir)
    elif os.path.isfile(pfname):
        return True
    else:
        return False
    
def readpick(name):
    pfname = pickle_dir + name + '_pickle.p'
    with open(pfname, 'rb') as pick:
        print('Reading pickle: '+name)
        pick_payload = pickle.load(pick)
        pick.close()
        return pick_payload
    
def writepick(name, pick_payload):
    pfname = pickle_dir + name + '_pickle.p'
    with open(pfname,'wb') as pf:   
        pprotocol = 2
        pickle.dump(pick_payload, pf, protocol=pprotocol)
        pf.close()
    print('couldnt write pickle file')
        
if __name__=='__main__':
    step00()
    step01()
    step02()
    
    
