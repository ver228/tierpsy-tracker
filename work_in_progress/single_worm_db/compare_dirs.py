# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 18:00:09 2016

@author: ajaver
"""

import os
def waldir(root_dir):
    all_files = []
    for dpath, dnames, fnames in os.walk(root_dir):
        for fname in fnames:
            fullfilename = os.path.abspath(os.path.join(dpath, fname))
            all_files.append(fullfilename)
        print(len(all_files))
    return all_files

if __name__ == '__main__':
    #dir2keep = '/Volumes/behavgenom_archive$/thecus/nas207-3/Data/from pc207-10/laura'
    #dir2del = '/Volumes/behavgenom_archive$/thecus/nas207-3/Data/from pc207-10/from pc207-10/laura'
    
    #dir2keep = '/Volumes/behavgenom_archive$/thecus/nas207-3/Data/from pc207-13/Laura/'
    #dir2del = '/Volumes/behavgenom_archive$/thecus/nas207-3/Data/from pc207-13/from pc207-13/Laura/'
    
    #dir2keep = '/Volumes/behavgenom_archive$/thecus/nas207-1/experimentBackup/from pc207-7/!worm_videos/copied_from_pc207-8/'    
    #dir2del = '/Volumes/behavgenom_archive$/thecus/nas207-1/experimentBackup/from pc220-6/!worm_videos/from pc207-8/'
    
    dir2keep = '/Volumes/behavgenom_archive$/thecus/nas207-1/experimentBackup/from pc220-6/!worm_videos/from pc207-15/misc_videos'
    dir2del = '/Volumes/behavgenom_archive$/thecus/nas207-1/experimentBackup/from pc207-18/!worm_videos/copied from pc207-15/misc_videos'
    
    #dir2del = '/Volumes/behavgenom_archive$/thecus/nas207-3/Data/from pc207-14/Laura/Analysis/analysis 19-08-09/'
    #dir2keep = '/Volumes/behavgenom_archive$/thecus/nas207-3/Data/from pc207-10/laura/analysis/analysis 19-08-09/'
    
    
    files2keep = waldir(dir2keep)
    files2del = waldir(dir2del)
    #%%
    files2del2 = [x for x in files2del if '.avi' in x]
    #files2del = [x.replace('.data/', '') for x in files2del if not '/normalized/' in x]

#%%
    for f2del in files2del:
        dd = f2del.replace(dir2del,dir2keep)
        if not dd in files2keep:
            print(f2del)
            #print("'" + f2del + "'", "'" + dd + "'")
        else:
            pass
            if os.path.getsize(f2del) > os.path.getsize(dd):                
                print(f2del)
                print(os.path.getsize(f2del) > os.path.getsize(dd))
                