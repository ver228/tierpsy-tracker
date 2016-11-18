# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 00:26:10 2016

@author: ajaver
"""
import os
import time
import datetime
import shutil

from MWTracker.helperFunctions.miscFun import print_flush
from MWTracker.batchProcessing.AnalysisPoints import AnalysisPoints
from MWTracker.batchProcessing.ProcessWormsWorker import ProcessWormsWorkerParser, ProcessWormsWorker, SCRIPT_WORKER
from MWTracker.batchProcessing.batchProcHelperFunc import create_script, getRealPathName

SCRIPT_LOCAL = getRealPathName(__file__)

class ProcessWormsLocal(object):
    def __init__(self, main_file, masks_dir, results_dir, tmp_mask_dir='',
            tmp_results_dir='', json_file='', analysis_checkpoints = [], 
            is_single_worm=False, use_skel_filter=True, is_copy_video = False):
        
        assert os.path.exists(main_file)
        self.main_file = main_file
        self.results_dir = results_dir
        self.masks_dir = masks_dir
        self.analysis_checkpoints = analysis_checkpoints
        
        self.json_file = json_file
        self.is_single_worm = is_single_worm
        self.use_skel_filter = use_skel_filter
        
        #we have both a mask and a results tmp directory because like that it is easy to asign them to the original if the are empty
        self.tmp_results_dir = tmp_results_dir if tmp_results_dir else results_dir
        self.tmp_mask_dir = tmp_mask_dir if tmp_mask_dir else masks_dir
        
        #we change the name of the main_file to the tmp directory if the is_copy_video is set to true
        #This flag should be optional in compress mode but true in track
        if is_copy_video:
            self.tmp_main_file = os.path.join(tmp_mask_dir, os.path.split(self.main_file)[1])
        else:
            self.tmp_main_file = self.main_file
            
        #make directories
        for dirname in [self.tmp_results_dir, self.tmp_mask_dir, self.results_dir, self.masks_dir]:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        
        #create objects for analysis points using the source and the temporary directory
        self.ap_src = AnalysisPoints(self.main_file, self.masks_dir, results_dir, self.json_file,
                 self.is_single_worm, self.use_skel_filter)
        self.base_name = self.ap_src.file_names['base_name']
        self.ap_tmp = AnalysisPoints(self.tmp_main_file, self.tmp_mask_dir, self.tmp_results_dir, self.json_file,
                 self.is_single_worm, self.use_skel_filter)
        
        #find the progress time in the temporary and source directory
        self.unfinished_points_src = self.ap_src.getUnfinishedPoints(self.analysis_checkpoints)
        self.unfinished_points_tmp = self.ap_tmp.getUnfinishedPoints(self.analysis_checkpoints)
        
        #get the points to be processed compared with the existing files
        self.checkpoints2process = self._getPoints2Process()
    
    # we need to group steps into start and clean steps for the multiprocess
    # part
    def start(self):
        
        self.start_time = time.time()

        #copy tmp files
        self._copyFinaltoTmp()
        
        args = [self.tmp_main_file]
        argkws = {'masks_dir':self.tmp_mask_dir, 'results_dir':self.tmp_results_dir, 
            'json_file':self.json_file, 'analysis_checkpoints':self.checkpoints2process,
            'is_single_worm':self.is_single_worm, 'use_skel_filter':self.use_skel_filter}

        return create_script(SCRIPT_WORKER, args, argkws)

    def clean(self):
        self._copyTmpToFinalAndClean()
        
        delta_t = time.time() - self.start_time
        time_str = datetime.timedelta(seconds = round(delta_t))
        
        progress_str = '{}  Finished. Total time {}.'.format(self.base_name, time_str)
        if len(self.unfinished_points_tmp) > 0:
            progress_str = '{} Missing analysis points in the tmp dir: {}.'.format(progress_str, self.unfinished_points_tmp)
        elif len(self.unfinished_points_src) > 0:
            progress_str = '''{} Missing analysis points in the final dir: {}. 
            Problems when copy files?.'''.format(progress_str, self.unfinished_points_src)
        
        print_flush(progress_str)
        
    def _getPoints2Process(self):
        def assignAndCheckSubset(small_list, larger_list):
            assert set(small_list).issubset(set(larger_list))
            return small_list
        
        if len(self.unfinished_points_src) < len(self.unfinished_points_tmp):
            checkpoints2process = assignAndCheckSubset(self.unfinished_points_src, self.unfinished_points_tmp)
        else:
            checkpoints2process = assignAndCheckSubset(self.unfinished_points_tmp, self.unfinished_points_src)
        return checkpoints2process
    
    def _copyFinaltoTmp(self):
        #files that are required as input
        inputs_required = self._points2Files(self.checkpoints2process, self.ap_tmp, "input_files")
        
        new_created_files = self._getNewFilesCreated(self.checkpoints2process, self.ap_tmp) 
        #files that are required as input but are not produced later on
        needed_files = inputs_required - new_created_files
        
        #files that will be created
        outputs_to_create = self._points2Files(self.checkpoints2process, self.ap_tmp, "output_files")



        #files from steps finished in the source but not in the tmp
        files_finished_src_no_tmp = self._getMissingFiles(self.unfinished_points_tmp, 
                                                           self.unfinished_points_src, 
                                                           self.ap_src)
        finished_points_tmp = set(self.analysis_checkpoints) - set(self.unfinished_points_tmp)
        
        input_finished_tmp = self._points2Files(finished_points_tmp, self.ap_tmp, "input_files")
        output_finished_tmp = self._points2Files(finished_points_tmp, self.ap_tmp, "output_files")
        file_finished_tmp = input_finished_tmp | output_finished_tmp
        
        #remove from list tmp files that are not in a later step in source
        filesnames2copy = needed_files - ( file_finished_tmp - files_finished_src_no_tmp) 
        
        files2copy = self._getFilesSrcDstPairs(filesnames2copy, 
                                               self.ap_src.file2dir_dict, 
                                               self.ap_tmp.file2dir_dict)
        self._copyFilesLocal(files2copy)
    
    def _copyTmpToFinalAndClean(self):
        '''copy files to final directory and clean'''
        
        #recalcuate the unfinished points in the tmp directory. At this point they should be empty
        self.unfinished_points_tmp = self.ap_tmp.getUnfinishedPoints(self.unfinished_points_src)
        if len(self.unfinished_points_tmp) != 0:
            #return, do not copy the files if the tmp analysis was interrupted
            return

        #copy all the files produced by the temp dir into the final destination
        ouput_files_produced = self._points2Files(self.unfinished_points_src, self.ap_tmp, "output_files")
        files2copy = self._getFilesSrcDstPairs(ouput_files_produced, 
                                               self.ap_tmp.file2dir_dict, 
                                               self.ap_src.file2dir_dict)
        self._copyFilesLocal(files2copy)

        self._deleteTmpFiles()
        

             

    def _deleteTmpFiles(self):    
        def _points2FullFiles(points2check, ap_obj, field_name):
            data = ap_obj.getField(field_name, points2check)
            return set(sum(data.values(), [])) #flatten list

        #check if the analysis was really finished in the src before deleting any files
        self.unfinished_points_src = self.ap_src.getUnfinishedPoints(self.analysis_checkpoints)
        if len(self.unfinished_points_src) != 0:
            #return, there is something weird with the source, it seems that the tmp files were not copy correctly
            return   
        
        #CLEAN
        all_tmp_files = _points2FullFiles(self.analysis_checkpoints, self.ap_tmp, "output_files") | \
        _points2FullFiles(self.analysis_checkpoints, self.ap_tmp, "input_files")
        
        all_src_files = _points2FullFiles(self.analysis_checkpoints, self.ap_src, "output_files") | \
        _points2FullFiles(self.analysis_checkpoints, self.ap_src, "input_files")
        
        #remove all tmp files that are not in the source
        files2remove = all_tmp_files - all_src_files

        for fname in files2remove:
            if os.path.exists(fname):
                os.remove(fname)
    
        
    def _getNewFilesCreated(self, points2process, ap_obj):
        '''get the new files that will be created after a given list of analysis points'''
        
        new_files = set()
        for ii in range(0, len(points2process)):
            current_point = [points2process[ii]]
            previous_points = points2process[:ii+1]
            output_files = set(self._points2Files(current_point, ap_obj, "output_files"))
            intput_files = set(self._points2Files(previous_points, ap_obj, "input_files"))
            
            #i want files that are outputs but where not inputs before
            new_files = new_files | (output_files - (output_files & intput_files))
        return new_files

    def _getMissingFiles(self, more_unfinished, less_unfinished, ap_obj_less):
        '''missing finished files after comparing two analysis (source and destination)'''
        points_missing = set(more_unfinished) - set(less_unfinished)
        return self._points2Files(points_missing, ap_obj_less, "output_files")
    
    def _points2Files(self, points_finished, ap_obj, field_name):
        '''convert analysis points to individual file names'''
        data = ap_obj.getField(field_name, points_finished)
        data = sum(data.values(), []) #flatten list
        data = [os.path.split(x)[1] for x in data] #split and only get the file names
        return set(data)

    def _getFilesSrcDstPairs(self, fnames, f2d_src, f2dir_dst):
        '''get the pair source file to destination directory'''
        return [(os.path.join(f2d_src[x], x), f2dir_dst[x]) for x in fnames]

    def _copyFilesLocal(self, files2copy):
        ''' copy files to the source directory'''
        for files in files2copy:
            file_name, destination = files
            assert(os.path.exists(destination))
    
            if os.path.abspath(os.path.dirname(file_name)
                               ) != os.path.abspath(destination):
                print_flush('Copying %s to %s' % (file_name, destination))
                shutil.copy(file_name, destination)


class ProcessWormsLocalParser(ProcessWormsLocal, ProcessWormsWorkerParser):
    
    def __init__(self, sys_argv):
        ProcessWormsWorkerParser.__init__(self)
        self.add_argument(
            '--tmp_mask_dir',
            default='',
            help='Temporary directory where the masked file is stored')
        self.add_argument(
            '--tmp_results_dir',
            default='',
            help='temporary directory where the results are stored')
        self.add_argument(
            '--is_copy_video',
            action='store_true',
            help='The video file would be copied to the temporary directory.')
        
        args = self.parse_args(sys_argv[1:])
        print(args)
        ProcessWormsLocal.__init__(self,
            **vars(args))


if __name__ == '__main__':
    import subprocess
    import sys
    
    #the local parser copy the tmp files into the tmp directory using its method start, and returns the command for the worm worker
    local_obj = ProcessWormsLocalParser(sys.argv)
    worker_cmd = local_obj.start()
    worm_parser = ProcessWormsWorkerParser()
    args = vars(worm_parser.parse_args(worker_cmd[2:]))
    ProcessWormsWorker(**args, cmd_original = subprocess.list2cmdline(sys.argv))

    #clear temporary files
    local_obj.clean()    