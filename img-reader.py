# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 15:25:28 2014

@author: richard

given an (x,y) return intensity value
"""


from PIL import Image
#from glob import glob
#import numpy as np

def pixel_intensity(coord, imgdir): #coord is (x,y)
    img =Image.open(imgdir).convert('L')
    return img.getpixel(coord)

#img =Image.open("vid1-000.png").convert('L')
#files = []
#for filename in glob('./vid1/*'):
#    files.append(filename)
#
#filesenum =list(enumerate(files))