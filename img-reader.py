# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 15:25:28 2014

@author: richard
"""

#v1
#from PIL import Image
#import numpy as np
#im=Image.open("vid1-000.png")
#pixels=np.asarray(im.getdata())
#npixels,bpp=pixels.shape
#
#with open('out.txt', 'w') as f:
#    f.write(pixels)


## v2
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import numpy as np
#
#imgasarray=mpimg.imread('vid1-000.png')
#imgasarray.tofile('arrayfile.txt')
##np.savetxt('test.txt', imgasarray)

from PIL import Image
from glob import glob
import numpy as np
img =Image.open("vid1-000.png").convert('L')
img.getpixel((60,0))

for filename in glob('./vid1/*'):
    print filename