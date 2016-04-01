# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:26:54 2016

@author: ajaver
"""

import numpy as np
from MWTracker.featuresAnalysis.obtainFeaturesHelper import _featureStat


feat = np.array((2,1,1))
modes = np.array((1,1,1))


#p = wormStatsClass()
stats = _featureStat(np.mean, feat, 'length', True, False, modes)


#