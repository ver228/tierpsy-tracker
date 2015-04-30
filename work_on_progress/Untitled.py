# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:46:22 2015

@author: ajaver
"""
#%%
import matplotlib.pylab as plt

plt.figure()

plt.subplot(2,2,1)
plt.plot(all_min_diff['HH_pos'])
plt.plot(all_min_diff['HT_pos'])

plt.subplot(2,2,2)
plt.plot(all_min_diff['TT_pos'])
plt.plot(all_min_diff['TH_pos'])

plt.subplot(2,2,3)
plt.plot(all_min_diff['HH_neg'])
plt.plot(all_min_diff['HT_neg'])

plt.subplot(2,2,4)
plt.plot(all_min_diff['TT_neg'])
plt.plot(all_min_diff['TH_neg'])

