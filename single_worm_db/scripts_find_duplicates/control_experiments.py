# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:45:16 2016

@author: ajaver
"""

#import pymysql.cursors

## Connect to the database
#connection = pymysql.connect(host='localhost',
#                             user='ajaver',
#                             db='worm_db',
#                             charset='utf8mb4',
#                             cursorclass=pymysql.cursors.DictCursor)
#
#
#with connection.cursor() as cursor:
#    sql = "SELECT * FROM `experiments`"
#    cursor.execute(sql)
#    result = cursor.fetchall()



#%%
import os
from glob import glob
import sys
from MWTracker.compressVideos.getAdditionalData import getAdditionalFiles
    

with open('single_worm_movies.txt', 'r') as fid:
    full_movie_files = fid.read().split('\n')


good_movies = []
seg_movies = []

for ii, mf in enumerate(full_movie_files):
    print(ii, len(full_movie_files))
    if not '_seg.avi' in mf and os.path.exists(mf):
        good_movies.append(mf)
    else:
        seg_movies.append(mf)
full_movie_files = good_movies


#%%
if False:
    from MWTracker.compressVideos.compressVideo import selectVideoReader
    
    corrupt_videos = []
    for ii, mf in enumerate(full_movie_files):
        print(ii, len(full_movie_files))
        vid, im_width, im_height, reader_type = selectVideoReader(mf)
        if im_width == 0 or im_height == 0:
            corrupt_videos.append(mf)

#%%
#dd = [x for x in errors_files if isinstance(x,FileNotFoundError)]
#for x in dd: print(x)   
#%%
directories, movie_files = zip(*map(os.path.split, full_movie_files))
#%%
dir_dict = {x:[] for x in set(movie_files)}

for mf, md in zip(movie_files, directories):
    dir_dict[mf].append(md)

duplicated_files = [x for x in dir_dict if len(dir_dict[x])>1]


print(len(dir_dict))
print(len(duplicated_files))

for ii in range(len(duplicated_files)):
    print(dir_dict[duplicated_files[ii]], duplicated_files[ii])

#%%
if True:
    
    additional_files = []
    no_addfiles = []
    empty_addfiles = []
    errors_files = []
    for ii, mf in enumerate(full_movie_files):
        print(ii, len(full_movie_files))
        #check for the additional files in the case of single worm
        try:
            #this function will throw and error if the .info.xml or .log.csv are not found
            info_file, stage_file = getAdditionalFiles(mf)
            additional_files.append((mf, (info_file, stage_file)))
        except (FileNotFoundError) as e:
            errors_files.append(e)
            no_addfiles.append(mf)
        except (IOError) as e:
            errors_files.append(e)
            empty_addfiles.append(mf)

    no_extra_files_full = (no_addfiles + empty_addfiles);
    bad_duplicates = {}
    for fname_full in no_extra_files_full:
        fdir, fname = os.path.split(fname_full)
        if fname in duplicated_files:
            if not fname in bad_duplicates: 
                bad_duplicates[fname] = []
            bad_duplicates[fname].append(fname_full)


#%%
bad_dirs = []
for x in duplicated_files:
    for dd in dir_dict[x]:
        flist = os.listdir(dd)
        flist = [m for m in flist if '.avi' in m]
        
        if all(m in duplicated_files for m in flist) and not dd in bad_dirs:
            bad_dirs.append(dd)

f2chosedir = []
for x in duplicated_files:
    if all(dd in bad_dirs for dd in dir_dict[x]):
        f2chosedir.append(x)

#%%
dirNoExtra = []
for x in f2chosedir:
    for dd in dir_dict[x]:
        try:
            info_file, stage_file = getAdditionalFiles(os.path.join(dd, x))
        except (FileNotFoundError, IOError):
            if not dd in dirNoExtra:
                dirNoExtra.append(dd)

assert len(dirNoExtra) == 0
        

#%%
#correct for wrongly assinged files
#%%
def getExtraFiles(video_name, video_dir):
    
    strdate = '__'.join(video_name.partition('_')[2].split('__')[0:2])
    
    opt_dir1 = glob(video_dir + os.sep + '.data' + os.sep + '*' + strdate + '*')
    opt_dir1 = [x for x in opt_dir1 if not '_seg.avi' in x]
    opt_dir2 = glob(video_dir + os.sep + '*' + strdate + '*')
    opt_dir2 = [x for x in opt_dir2 if not '_seg.avi' in x]
    if not (len(opt_dir1) <= 1 or len(opt_dir2) <= 1):
        print(opt_dir1)
        print(opt_dir2)
        raise
    
    otherfiles = opt_dir1 if len(opt_dir1)>len(opt_dir2) else opt_dir2
    
    otherfiles = [x for x in otherfiles if not '.avi' in x]
    assert len(otherfiles) != 1
    
    return otherfiles
#%% Correct names
if False:
    for x in no_addfiles:
        dname, fname = os.path.split(x)
        #print(fname)
        
        extra_log_files = getExtraFiles(fname, dname) 
        n_info_files = sum(x.endswith('.info.xml') for x in extra_log_files)
        n_csv_files = sum(x.endswith('.log.csv') for x in extra_log_files)
        
        
        fprefix = fname.split('.')[0]
        
        assert n_info_files<=1 and n_csv_files<=1
        
        if n_info_files == 1 and n_info_files == 1:
            for extra_x in extra_log_files:
                xtradir, xtrafname = os.path.split(extra_x)
                xtraprefix, _, xtraext = xtrafname.partition('.')
                if xtraprefix != fprefix:
                    newname = xtradir + os.sep + fprefix + '.' + xtraext
                    os.rename(extra_x, newname)
                    print(newname)

#%%

with open('single_worm_movies_filter.txt', 'w') as fid:
    for fname in full_movie_files:
        #dname =  dir_dict[fname]
        #assert len(dname) == 1
        fid.write(fname + '\n')