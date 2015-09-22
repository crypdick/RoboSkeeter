# -*- coding: utf-8 -*-
"""
Correlation analysis of mosquito flight

Created on Thu May 14 13:12:15 2015

@author: richard
"""

import os

import statsmodels.tsa
import statsmodels.graphics.tsaplots
import matplotlib.pyplot as plt

from scripts import io

plt.style.use('ggplot')

def make_csv_name_list():
    # TODO export this to io

    print "Loading + filtering CSV files from ", TRAJECTORY_DATA_DIR
    os.chdir(TRAJECTORY_DATA_DIR)
    csv_list = sorted([os.path.splitext(file)[0] for file in glob("*.csv")])
    os.chdir(os.path.dirname(__file__))  # go back to old dir

    return csv_list

csv_list = make_csv_name_list()

#make_csv_name_list()

for csv_fname in csv_list:
    print csv_fname
    os.chdir(os.path.dirname(__file__))
    df = io.load_csv2DF(csv_fname)

    if not os.path.exists('./correlation_figs/{data_name}'.format(data_name = csv_fname)):
        os.makedirs('./correlation_figs/{data_name}'.format(data_name = csv_fname))
    
    
    for label, col in df.iteritems():
    #    kwargs = {'title': 'Autocorrelation %s' % label}
        print label
        
        acf_fig = statsmodels.graphics.tsaplots.plot_acf(col, lags = 70)
        plt.savefig("./correlation_figs/{data_name}/{label} - ACF.svg".format(label=label, data_name = csv_fname), format="svg")
        plt.show()
        
        pacf_fig = statsmodels.graphics.tsaplots.plot_pacf(col, lags = 70, method='ywmle')
        plt.savefig("./correlation_figs/{data_name}/{label} - PACF.svg".format(label=label, data_name = csv_fname), format="svg")
        plt.show()
    
        
"""
add_subplot
"""
