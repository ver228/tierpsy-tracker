# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 00:26:10 2016

@author: ajaver
"""
import argparse
import datetime
import os
import time

from tierpsy.helper.misc import print_flush
from tierpsy.processing.AnalysisPoints import AnalysisPoints
from tierpsy.processing.trackProvenance import getPackagesVersion, execThisPoint
from tierpsy.processing.helper import get_real_script_path


BATCH_SCRIPT_WORKER = get_real_script_path(__file__, 'ProcessWorker')

class ProcessWorker(object):
    def __init__(self, 
                main_file, 
                masks_dir, 
                results_dir, 
                json_file, 
                analysis_checkpoints, 
                cmd_original,
                overwrite_prev = False):
        
        for dirname in [masks_dir, results_dir]:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        
        
        self.ap = AnalysisPoints(main_file, masks_dir, results_dir, json_file)
        self.analysis_checkpoints = self.ap.getUnfinishedPoints(analysis_checkpoints)
        self.cmd_original = cmd_original
        self.execAllPoints()
    
    def execAllPoints(self):
        base_name = self.ap.file_names['base_name']
        if len(self.analysis_checkpoints) == 0:
            print_flush('%s No checkpoints given. It seems that there is a previous analysis finished. Exiting.' % base_name)
            return
        
        pkgs_versions = getPackagesVersion()
        
        print_flush('%s Starting checkpoint: %s' % (base_name, self.analysis_checkpoints[0]))
        initial_time = time.time()
        for current_point in self.analysis_checkpoints:
            print(current_point)
            unmet_requirements = self.ap.hasRequirements(current_point)

            if len(unmet_requirements) != 0:
                print(unmet_requirements)
                break
            
            this_point_exists = self.ap.checker.get(current_point)
            if this_point_exists:
                print('this_point_exists', current_point)
                break
            
            execThisPoint(current_point, 
                    **self.ap.getArgs(current_point),
                    pkgs_versions = pkgs_versions,
                    cmd_original = self.cmd_original)
    
        time_str = str(datetime.timedelta(seconds = round(time.time() -initial_time)))
        
        if len(unmet_requirements) > 0:
            print_flush('''{} Finished early. Total time {}. Cannot continue 
            for step {} because it does not sastify the requiriments: {}'''.format(
            base_name, time_str, current_point, unmet_requirements))
        elif this_point_exists:
            existing_files = self.ap.getField('output_files', [current_point]).values()
            print_flush('''{} Finished early. Total time {}. the step {} 
            already exists. Delete files if you want to continue: 
            {}'''.format(base_name, time_str, current_point, existing_files))
        else:
            print_flush('{}  Finished in {}. Total time {}.'.format(base_name, current_point, time_str))
        


class ProcessWorkerParser(argparse.ArgumentParser):
    def __init__(self):
        super(ProcessWorkerParser, self).__init__()
        self.add_argument('main_file', help='hdf5 or video file.')
        self.add_argument('--masks_dir',
            help='Directory where the masked images are saved or are going to be saved')
        self.add_argument('--results_dir', 
            help='Directory where the trackings results are going to be saved.')
        self.add_argument(
            '--json_file',
            default='',
            help='File (.json) containing the tracking parameters.')
        self.add_argument(
            '--analysis_checkpoints', nargs='*',
            help='Sequence of steps to be processed.')
        


if __name__ == '__main__':
    import sys
    import subprocess
    
    worm_parser = ProcessWorkerParser()
    args = vars(worm_parser.parse_args())
    ProcessWorker(**args, cmd_original = subprocess.list2cmdline(sys.argv))
