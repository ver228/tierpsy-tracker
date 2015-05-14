# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:59:56 2015

@author: ajaver
"""

#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import pstats, cProfile

import fixHTorientation_tables
#import re
#cProfile.run('re.compile("foo|bar")')

cProfile.runctx("fixHTorientation_tables.main()", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()