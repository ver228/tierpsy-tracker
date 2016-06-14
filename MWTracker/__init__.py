# # -*- coding: utf-8 -*-
# """
# Created on Tue Jul  7 11:29:01 2015

# @author: ajaver
# """

from .version import __version__

#force to use matplotlib with qt5
import matplotlib
matplotlib.use('Qt5Agg', force=True)