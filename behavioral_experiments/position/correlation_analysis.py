# -*- coding: utf-8 -*-
"""
Correlation analysis of mosquito flight

Created on Thu May 14 13:12:15 2015

@author: richard
"""

import trajectory_data_io
import statsmodels.tsa
import statsmodels.graphics.tsaplots
import matplotlib.pyplot as plt
plt.style.use('ggplot')


df, data_name = trajectory_data_io.load_trajectory_dynamics_csv('rightplumesample2')
#velo_y_ts = df.velo_y

if not os.path.exists('./correlation_figs/{data_name}'.format(data_name = data_name)):
    os.makedirs('./correlation_figs/{data_name}'.format(data_name = data_name))


for label, col in df.iteritems():
#    kwargs = {'title': 'Autocorrelation %s' % label}
    print label
    acf_fig = statsmodels.graphics.tsaplots.plot_acf(col, lags = 70)
    plt.savefig("./correlation_figs/{data_name}/{label} - ACF.svg".format(label=label, data_name = data_name), format="svg")
    plt.show()
    
    pacf_fig = statsmodels.graphics.tsaplots.plot_pacf(col, lags = 70)
    plt.savefig("./correlation_figs/{data_name}/{label} - PACF.svg".format(label=label, data_name = data_name), format="svg")
    plt.show()