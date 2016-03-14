# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 19:25:53 2016

@author: ajaver
"""
import os
import fnmatch
import shutil
    
def getAllFiles():
    root_dir = '/Volumes/behavgenom_archive$/thecus/'
    root_dir = os.path.abspath(root_dir)
    assert os.path.exists(root_dir)
    	
    pattern_include = '*.avi'
    #%%
    tot = 0;
    with open('single_worm_files.txt', 'w') as fid:
        valid_files = []
        for dpath, dnames, fnames in os.walk(root_dir):
            for fname in fnames:
                if fnmatch.fnmatch(fname, pattern_include):
                    fullfilename = os.path.abspath(os.path.join(dpath, fname))
                    assert os.path.exists(fullfilename)
                    valid_files.append(fullfilename)
                    #print(fullfilename)
                    fid.write(fullfilename + '\n')
                    
                    tot += 1
                    print(tot)

if __name__ == '__main__':
    getAllFiles()
    
    
    what
    with open('single_worm_files.txt', 'r') as fid:
        valid_files = fid.read().split('\n')


    str2find = ['W03B1.2 (ok2433) on food L_2010_04_21__11_50_54__5',
                'W03B1.2 (ok2433) on food R_2010_04_22__16_02_55__15',
                'dpy-20 (e1282)IV on food R_2011_08_04__11_09_01___1___3',
                'dpy-20 (e1282)IV on food L_2011_08_09__12_33_49__7',
                '972 JU345 on food L_2011_02_24__14_53___3___3',
                '972 JU345 on food L_2011_03_17__16_28_09___2___11',
                'unc-7 (cb5) on food L_2010_05_20__12_54_20___2___6',
                'unc-7 (cb5) on food L_2010_08_19__11_20_42__3',
                'Mec-12 (u76) on food L_2010_11_11__11_30_42___7___2',
                'mec-12 (u76) on food L_2010_10_21__12_48_43__6']
    #str2find = ['W03B1.2 (ok2433)', 'dpy-20 (e1282)IV', 'mec-12 (u76)', 'unc-7 (cb5)' ,'JU345']

    files2check = {}
    for ftype in str2find:
        files2check[ftype] = [x for x in valid_files if ftype.lower() in x.lower() and not '_seg.avi' in x.lower() and 'on food' in x.lower()]
    print({x:len(files2check[x]) for x in files2check})
    #%%

    destination_dir = '/Users/ajaver/Desktop/Videos/single_worm/Teodor/'
    for mm in files2check:
        for ff in files2check[mm]:
            ff_dir, ff_file = os.path.split(ff)
            try:            
                aux1 = os.path.join(ff_dir, '.data', ff_file.replace('.avi', '.log.csv'))
                aux2 = os.path.join(ff_dir, '.data', ff_file.replace('.avi', '.info.xml'))
                shutil.copy(aux1, destination_dir)
                shutil.copy(aux2, destination_dir)
            except FileNotFoundError:
                try:
                    aux1 = os.path.join(ff_dir,  ff_file.replace('.avi', '.log.csv'))
                    aux2 = os.path.join(ff_dir,  ff_file.replace('.avi', '.info.xml'))
                    shutil.copy(aux1, destination_dir)
                    shutil.copy(aux2, destination_dir)
                except FileNotFoundError:
                    continue

