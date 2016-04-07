# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 22:01:59 2016

@author: ajaver
"""
import sys
import os
import h5py
import subprocess as sp
from threading  import Thread
from queue import Queue, Empty
import shutil
import argparse
import tempfile

from MWTracker.helperFunctions.runMultiCMD import runMultiCMD, print_cmd_list
from MWTracker.helperFunctions.miscFun import print_flush

class alignSingleLocal:
    def __init__(self, dat):
        masked_image_file, skeletons_file, tmp_dir = dat

        self.masked_image_file = os.path.abspath(masked_image_file)
        self.skeletons_file = os.path.abspath(skeletons_file)

        self.base_name = os.path.split(masked_image_file)[1].rpartition('.')[0];

        #here i am using the same directory for everything, be sure there are not files with the same name.
        
        self.tmp_dir = tmp_dir;
        if len(self.tmp_dir)>0:
            self.tmp_dir = os.path.abspath(self.tmp_dir) 

            if not os.path.exists(self.tmp_dir):
                os.makedirs(self.tmp_dir)
            self.masked_image_tmp = os.path.join(tmp_dir, os.path.split(masked_image_file)[1])
            self.skeletons_tmp = os.path.join(tmp_dir, os.path.split(skeletons_file)[1])
        else:
            self.masked_image_tmp = self.masked_image_file
            self.skeletons_tmp = self.skeletons_file


    def start(self):
        
        if os.path.abspath(self.masked_image_tmp) != os.path.abspath(self.masked_image_file):
            print(self.base_name + ' Copying mask file to temporary directory.')
            shutil.copy(self.masked_image_file, self.masked_image_tmp)
            assert os.path.exists(self.masked_image_tmp)

        if os.path.abspath(self.skeletons_tmp) != os.path.abspath(self.skeletons_file):
            print(self.base_name + ' Copying skeletons file to temporary directory.')
            shutil.copy(self.skeletons_file, self.skeletons_tmp)
            assert os.path.exists(self.skeletons_tmp)
        return self.create_script()

        print(self.base_name + ' Starting aligment.')

    def create_script(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        start_cmd = '/Applications/MATLAB_R2014b.app/bin/matlab -nojvm -nosplash -nodisplay -nodesktop <'.split()
        
        script_cmd = "addpath('{0}'); " \
        "try, alignStageMotionSegwormFun('{1}', '{2}'); " \
        "catch ME, disp(getReport(ME)); " \
        "end; exit; "
        script_cmd = script_cmd.format(current_dir, self.masked_image_tmp, self.skeletons_tmp)

        self.tmp_fid, self.tmp_script_file = tempfile.mkstemp(suffix='.m', dir=self.tmp_dir, text=True)
        with open(self.tmp_script_file, 'w') as fid:
            fid.write(script_cmd)

        matlab_cmd = start_cmd + [self.tmp_script_file]

        return matlab_cmd

    def clean(self):
        os.close(self.tmp_fid)
        os.remove(self.tmp_script_file)

        print(self.base_name + ' Deleting files to temporary files.')
        if os.path.abspath(self.skeletons_tmp) != os.path.abspath(self.skeletons_file):
            shutil.copy(self.skeletons_tmp, self.skeletons_file)
            assert os.path.exists(self.skeletons_file)
            os.remove(self.skeletons_tmp)
        
        if os.path.abspath(self.masked_image_tmp) != os.path.abspath(self.masked_image_file):
            assert os.path.exists(self.masked_image_file)
            os.remove(self.masked_image_tmp)

        print(self.base_name + ' Finished.')


def main(mask_list_file, tmp_dir, max_num_process, refresh_time, reset) :
    
    with open(mask_list_file, 'r') as fid:
        mask_files = fid.read().split('\n')

    getSkelF = lambda  x: x.replace('MaskedVideos', 'Results').replace('.hdf5', '_skeletons.hdf5')
    originalFiles = [(x, getSkelF(x)) for x in mask_files if x]
    
    files2check = []
    for mask_file, skel_file in originalFiles:
        assert os.path.exists(mask_file)
        assert os.path.exists(skel_file)

        with h5py.File(skel_file, 'r+') as fid:
            try:
                if reset: fid['/stage_movement'].attrs['has_finished'] = 0
                has_finished = fid['/stage_movement'].attrs['has_finished'][:]
            except (KeyError,IndexError):
                has_finished = 0;

        if has_finished == 0:
            files2check.append(('', mask_file, skel_file, tmp_dir))


    #make sure all the files have unique names, otherwise having the temporary direcotry can cause problems 
    base_names = [os.path.split(x[1])[1] for x in files2check]
    assert len(base_names) == len(set(base_names))
    
    print('Files to be processed:', len(files2check))
    runMultiCMD(files2check, local_obj=alignSingleLocal, max_num_process = max_num_process, refresh_time = refresh_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Align stage motions with videos using matlab scripts.")
    
    parser.add_argument('mask_list_file', help='File containing the full path of the masked videos to be analyzed, otherwise there will be search from video_dir_root using pattern_include and pattern_exclude.')
    
    parser.add_argument('--tmp_dir', default = os.path.join(os.path.expanduser("~"), 'Tmp'), \
        help='Temporary directory where files are going to be stored. Make sure there are no files with duplicated names.')
    
    parser.add_argument('--max_num_process', default = 6, type = int, help = 'Max number of process to be executed in parallel.')
    parser.add_argument('--refresh_time', default = 10, type = float, help = 'Refresh time in seconds of the process screen.')
    parser.add_argument('--reset', action='store_true', help = 'Reset the has_finished flag of all the files to zero.')

    args = parser.parse_args()

    main(**vars(args))

        