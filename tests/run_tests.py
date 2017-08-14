
import sys
import os
import shutil
import stat
import glob
import json
import argparse

import tierpsy

class TestObj():
    def __init__(self, examples_dir, script_dir):
        self.commands = []
        self.process_script = os.path.join(script_dir, 'processMultipleFiles.py')
        self.main_dir = os.path.join(examples_dir, self.name)
        self.masked_files_dir = os.path.join(self.main_dir, 'MaskedVideos')
        self.raw_video_dir = os.path.join(self.main_dir, 'RawVideos')
        self.results_dir = os.path.join(self.main_dir, 'Results')

    def add_command(self, args, dlft_script=''):
        if not dlft_script:
            dlft_script = self.process_script

        command = [sys.executable, dlft_script] + args

        cmd_dd = []
        for ii, x in enumerate(command):
            if ii != 0 and not x.startswith('--'):
                cmd_dd.append('"' + x + '"')
            else:
                cmd_dd.append(x)

        cmd_dd = ' '.join(cmd_dd)
        self.commands.append(cmd_dd)

        return self.commands

    def execute_cmd(self):
        for cmd in self.commands:
            print(cmd)
            print('%%%%%% {} %%%%%%'.format(self.name))
            print(self.description)
            os.system(cmd)

    
    def remove_dir(self, dir2remove):
        if os.path.exists(dir2remove):
            if sys.platform == 'darwin':
                for fname in glob.glob(os.path.join(dir2remove, '*.hdf5')):
                    os.chflags(fname, not stat.UF_IMMUTABLE)
            shutil.rmtree(dir2remove)

    def run(self):
        self.clean()
        self.execute_cmd()

    def clean(self):
        self.remove_dir(self.masked_files_dir)
        self.remove_dir(self.results_dir)



class GECKO_VIDEOS(TestObj):
    def __init__(self, *args):
        self.name = 'GECKO_VIDEOS'
        self.description = 'Complete analysis from video from Gecko .mjpg files.'
        super().__init__(*args)

        
        args = [
        '--video_dir_root',
        self.raw_video_dir,
        '--mask_dir_root',
        self.masked_files_dir,
        '--results_dir_root',
        self.results_dir,
        '--pattern_include',
        '*.mjpg',
        '--json_file',
        'MULTI_RIG.json'
        ]
        self.add_command(args)

class AVI_VIDEOS(TestObj):
    def __init__(self, *args):
        self.name = 'AVI_VIDEOS'
        self.description = 'Complete analysis from .avi files.'
        super().__init__(*args)

        json_file = os.path.join(self.main_dir, 'AVI_VIDEOS.json')
        args = [
        '--video_dir_root',
        self.raw_video_dir,
        '--mask_dir_root',
        self.masked_files_dir,
        "--json_file",
        json_file,
        '--pattern_include',
        '*.avi'
        ]
        self.add_command(args)



class MANUAL_FEATS(TestObj):
    def __init__(self, *args):
        self.name = 'MANUAL_FEATS'
        self.description = 'Calculate features from manually joined trajectories.'
        super().__init__(*args)

        args = [
        '--mask_dir_root',
        self.masked_files_dir,
        '--analysis_checkpoints',
        'FEAT_MANUAL_CREATE',
        "--json_file",
        'MULTI_RIG.json'
        ]
        self.add_command(args)

    def clean(self):
        fnames = glob.glob(os.path.join(self.main_dir, 'Results', '*_feat_manual.hdf5'))
        for fname in fnames:
            os.remove(fname)


class RIG_HDF5_VIDEOS(TestObj):
    def __init__(self, *args):
        self.name = 'RIG_HDF5_VIDEOS'
        self.description = 'Reformat hdf5 file produced by the gecko plugin in the worm rig.'
        super().__init__(*args)

        args = [
        '--video_dir_root',
        self.raw_video_dir,
        '--mask_dir_root',
        self.masked_files_dir,
        '--pattern_include',
        '*.raw_hdf5',
        '--json_file',
        'MULTI_RIG.json'
        ]
        self.add_command(args)


class SCHAFER_LAB_SINGLE_WORM(TestObj):
    def __init__(self, *args):
        self.name = 'SCHAFER_LAB_SINGLE_WORM'
        self.description = "Schaffer's lab single worm tracker."
        super().__init__(*args)

        args = [
        '--video_dir_root',
        self.raw_video_dir,
        '--mask_dir_root',
        self.masked_files_dir,
        '--results_dir_root',
        self.results_dir,
        '--json_file',
        'SINGLE_WORM_SHAFER_clockwise.json',
        '--pattern_include', 
        '*.avi',
        ]
        self.add_command(args)


class WORM_MOTEL(TestObj):
    def __init__(self, *args):
        self.name = 'WORM_MOTEL'
        self.description = "Worm motel (background subtraction)."
        super().__init__(*args)

        args = [
        '--video_dir_root',
        self.raw_video_dir,
        '--mask_dir_root',
        self.masked_files_dir,
        '--results_dir_root',
        self.results_dir,
        '--json_file',
        'MULTI_RIG_worm_motel',
        '--pattern_include', 
        '*.mjpg'
        ]
        self.add_command(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('n_tests', metavar='N', type=int, nargs='*',
                        help='Number of tests to be done. If it is empty it will execute all the tests.')
    parser.add_argument('--tests', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')
    args = parser.parse_args()

    n_tests = args.n_tests
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(tierpsy.__file__), '..'))
    examples_dir = os.path.join(root_dir, 'tests', 'data')
    script_dir = os.path.join(root_dir, 'cmd_scripts')

    all_tests_obj = [
                    GECKO_VIDEOS, 
                    AVI_VIDEOS, 
                    MANUAL_FEATS, 
                    RIG_HDF5_VIDEOS, 
                    SCHAFER_LAB_SINGLE_WORM, 
                    WORM_MOTEL
                    ]
    all_tests = [obj(examples_dir, script_dir) for obj in all_tests_obj]

    tests_ind = [x-1 for x in n_tests]
    if tests_ind:
        test_to_exec = [all_tests[x] for x in tests_ind]
    else:
        test_to_exec = all_tests #execute all tests
    
    
    for test in test_to_exec:
        test.run()
