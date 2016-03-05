# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:10:03 2016

@author: ajaver
"""
import os

with open('single_worm_movies.txt', 'r') as fid:
    full_movie_files = fid.read().split('\n')
    full_movie_files = [x for x in full_movie_files if not x.endswith('_seg.avi')]
    full_movie_files = [x for x in full_movie_files if x]
    print(len(full_movie_files))

with open('stage_move_csv.txt', 'r') as fid:
    full_csv_files = fid.read().split('\n')
    full_csv_files = [x for x in full_csv_files if x]
    print(len(full_csv_files))

with open('video_info_xml.txt', 'r') as fid:
    full_xml_files = fid.read().split('\n')
    full_xml_files = [x for x in full_xml_files if x]
    print(len(full_xml_files))

#%%

all_files_d = {}

for fnamefull in full_movie_files:
    dname, fname = os.path.split(fnamefull);
    basename = fname[:-4]
    
    if not basename in all_files_d: 
        all_files_d[basename] = []
    all_files_d[basename].append(fnamefull)

for fnamefull in full_xml_files:
    dname, fname = os.path.split(fnamefull);
    basename = fname[:-9]
    
    if not basename in all_files_d:
        all_files_d[basename] = []
    all_files_d[basename].append(fnamefull)
    
for fnamefull in full_csv_files:
    dname, fname = os.path.split(fnamefull);
    basename = fname[:-8]
    
    if not basename in all_files_d:
        all_files_d[basename] = []
    all_files_d[basename].append(fnamefull)
#%%
    
#all_files_d[basename][0] = all_files_d[basename][0] + 'a'
#all_files_d[basename][1] = 'b ' + all_files_d[basename][1]

wrong_dir = {}
wrong_files = {}
for fbase in all_files_d:
    flist = all_files_d[fbase]
    if len(flist) == 3:
        dnames, fnames = zip(*map(os.path.split, flist))
        
        if not (sum(x.endswith('.avi') for x in fnames) == 1 and \
        sum(x.endswith('.log.csv') for x in fnames) == 1 and \
        sum(x.endswith('.info.xml') for x in fnames) == 1):
            wrong_files[fbase] = flist
            continue
        
        dnames = [x.replace('/.data', '') for x in dnames]
        if not (dnames[0] == dnames[1] == dnames[2]):
            wrong_dir[fbase] = flist
#%%
more_files = {x:all_files_d[x] for x in all_files_d if len(all_files_d[x]) > 3}

less_files = {x:all_files_d[x] for x in all_files_d if len(all_files_d[x]) < 3}

#%%

dates = {}
for x in less_files:
    datestr = '20' + x.partition('20')[-1]; #will fail sometimes but for this case might be fine
    if not datestr in dates:
        dates[datestr] = []
    dates[datestr].append(x)
maybetogether = [dates[x] for x in dates if len(dates[x]) > 1]

#%%
from collections import OrderedDict
remaining_dates = OrderedDict()

for x in sorted(dates):    
    if len(dates[x]) > 1: continue
    dd = x.partition('__')[0]
    if not dd in remaining_dates: remaining_dates[dd] = []
    remaining_dates[dd] += dates[x]
#%%
for x in remaining_dates:
    if len(remaining_dates[x])>1:
        print(x)
        for d in remaining_dates[x]:
            print(less_files[d])