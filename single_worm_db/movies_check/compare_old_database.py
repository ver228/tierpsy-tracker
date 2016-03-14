# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:45:16 2016

@author: ajaver
"""
import pymysql.cursors

import os
from glob import glob
import sys
sys.path.append('/Users/ajaver/Documents/GitHub/Multiworm_Tracking')
from MWTracker.compressVideos.getAdditionalData import getAdditionalFiles

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='ajaver',
                             db='worm_db',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

#sql_cmd = '''SELECT e.id, e.wormFileName FROM exp_annotation AS d JOIN  experiments AS e ON e.id = d.id;'''
sql_cmd = 'SELECT * FROM experiments'
with connection.cursor() as cursor:
    cursor.execute(sql_cmd)
    result = cursor.fetchall()

db_fnames = [x['wormFileName'] for x in result]
db_fnames = set(db_fnames)

#%%
with open('single_worm_movies.txt', 'r') as fid:
    full_movie_files = fid.read().split('\n')
    
full_movie_files = [x for x in full_movie_files if not '_seg.avi' in x and x]

directories, movie_files = zip(*map(os.path.split, full_movie_files))
loc_fnames = set([x[:-4] for x in movie_files])

notInLoc = db_fnames-loc_fnames
notInDB = loc_fnames-db_fnames

print('In the DB but not in Local: %i' % len(notInLoc))
print('Not DB but in Local: %i' % len(notInDB))
