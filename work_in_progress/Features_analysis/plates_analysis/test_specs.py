# -*- coding: utf-8 -*-

import os, sys
#import sys
import tables
import pandas as pd
import numpy as np

sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking')

import warnings
warnings.filterwarnings('ignore', '.*empty slice*',)
tables.parameters.MAX_COLUMNS = 1024 #(http://www.pytables.org/usersguide/parameter_files.html)

from collections import OrderedDict


from MWTracker import config_param
from movement_validation import WormFeatures, FeatureProcessingOptions

from MWTracker.featuresAnalysis.obtainFeaturesHelper import wormStatsClass, WormFromTable

