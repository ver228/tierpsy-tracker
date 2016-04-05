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

from MWTracker.helperFunctions.runMultiCMD import runMultiCMD, print_cmd_list
from MWTracker.helperFunctions.miscFun import print_flush

class alignSingleLocal:
    def __init__(self, dat):
        masked_image_file, skeletons_file, tmp_dir = dat

        self.masked_image_file = os.path.abspath(masked_image_file)
        self.skeletons_file = os.path.abspath(skeletons_file)

        self.base_name = os.path.split(masked_image_file)[1].rpartition('.')[0];

        #here i am using the same directory for everything, be sure there are not files with the same name.
        self.tmp_dir = os.path.abspath(tmp_dir) 
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        if self.tmp_dir:
            self.masked_image_tmp = os.path.join(tmp_dir, os.path.split(masked_image_file)[1])
            self.skeletons_tmp = os.path.join(tmp_dir, os.path.split(skeletons_file)[1])
        else:
            self.masked_image_tmp = self.masked_image_file
            self.skeletons_tmp = self.skeletons_file

    def start(self):
        print(self.base_name + ' Copying files to temporary directory.')
        if os.path.abspath(self.masked_image_tmp) != os.path.abspath(self.masked_image_file):
            shutil.copy(self.masked_image_file, self.masked_image_tmp)
            assert os.path.exists(self.masked_image_tmp)

        if os.path.abspath(self.skeletons_tmp) != os.path.abspath(self.skeletons_file):
            shutil.copy(self.skeletons_file, self.skeletons_tmp)
            assert os.path.exists(self.skeletons_tmp)
        return self.create_script()

    def create_script(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        start_cmd = '/Applications/MATLAB_R2014b.app/bin/matlab -nojvm -nosplash -nodisplay -nodesktop -r'.split()
        script_cmd = '''addpath('{0}'); alignStageMotionSegwormFun('{1}', '{2}');''' \
        ''' exit'''.format(current_dir, self.masked_image_tmp, self.skeletons_file)
        
        #script_cmd = "disp(\'hola\'); exit;"
        matlab_cmd = start_cmd + [script_cmd]

        return matlab_cmd

    def clean(self):
        print(self.base_name + ' Deleting files to temporary files.')
        if os.path.abspath(self.skeletons_tmp) != os.path.abspath(self.skeletons_file):
            shutil.copy(self.skeletons_tmp, self.skeletons_file)
            assert os.path.exists(self.skeletons_file)
            os.remove(self.skeletons_tmp)
        
        if os.path.abspath(self.masked_image_tmp) != os.path.abspath(self.masked_image_file):
            assert os.path.exists(self.masked_image_file)
            os.remove(self.masked_image_tmp)

        print(self.base_name + ' Finished.')


if __name__ == '__main__':
    #print(sys.argv)
    mask_list_file = sys.argv[1]
    max_num_process = 6
    refresh_time = 10
    tmp_dir = os.path.join(os.path.expanduser("~"), 'Tmp')

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
                #fid['/stage_movement'].attrs['has_finished']=0
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

#ON_POSIX = 'posix' in sys.builtin_module_names
#    for ii, (_, mask_file, skel_file, tmp_dir) in enumerate(files2check[:1]):
#          dd = alignSingleLocal([mask_file, skel_file, tmp_dir])
#          cmd = dd.start()
#          pid = sp.Popen(cmd, stdout = sp.PIPE, stderr = sp.PIPE,
#                            close_fds = ON_POSIX)
#          
          #sp.call(cmd)
          #outs, errs = pid.communicate(timeout=90)
          #dd.clean()

    #print(matlab_cmd)
    #sp.call(matlab_cmd, shell=True)


