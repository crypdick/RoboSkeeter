# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:09:36 2015

@author: fairhalladmin
"""

import optimizer

guess = None
score = 10000000

for i in range(1):
    print("iter {}".format(i))
    guess, score = optimizer.main(x_0=guess)
    print("guess: {}, score: {}".format(guess, score))
    
print guess, score