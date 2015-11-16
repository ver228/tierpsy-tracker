# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 23:46:32 2015

@author: ajaver
"""

import sys
sys.path.append('../../../movement_validation')

#from movement_validation.pre_features import WormParsing
from movement_validation.statistics.histogram_manager import HistogramManager
from movement_validation.statistics import specs
from movement_validation import utils


#class histTest(HistogramManager):
#    def __init__(self):
#        print('hola')
        

#specs.EventSpecs.getSpecs()

a = HistogramManager()
a.init_histograms(openworm_features)
#[utils.filter_non_numeric(spec) for spec in specs.MovementSpecs.getSpecs()]

#specs.SimpleSpecs.getSpecs()
#specs.EventSpecs.getSpecs()