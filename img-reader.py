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
#    print img.getpixel(coord)
    return img.getpixel(coord)

pixel_intensity((0,0), "./example_vids/fullstim.png")

def xycoord2pixcoord(xycoord):
    pass #need to get y value and take abs value

#img =Image.open("vid1-000.png").convert('L')
#files = []
#for filename in glob('./vid1/*'):
#    files.append(filename)
#
#filesenum =list(enumerate(files))