# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 15:25:28 2014

@author: richard
"""

#from PIL import Image
#import numpy as np
#im=Image.open("vid1-000.png")
#pixels=np.asarray(im.getdata())
#npixels,bpp=pixels.shape
#
#with open('out.txt', 'w') as f:
#    f.write(pixels)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

imgasarray=mpimg.imread('vid1-000.png')
imgasarray.tofile('arrayfile.txt')
#np.savetxt('test.txt', imgasarray)
