
import sys
import os
import shutil
import stat
import glob
import argparse

import tierpsy

class TestObj():
    def __init__(self, examples_dir, script_dir):
        self.process_script = os.path.join(script_dir, 'processMultipleFiles.py')
        self.main_dir = os.path.join(examples_dir, self.name)
        self.masked_files_dir = os.path.join(self.main_dir, 'MaskedVideos')
        self.raw_video_dir = os.path.join(self.main_dir, 'RawVideos')
        self.results_dir = os.path.join(self.main_dir, 'Results')


    def execute_cmd(self):
        self.command = [sys.executable, self.process_script] + self.args

        cmd_dd = []
        for ii, x in enumerate(self.command):
            if ii != 0 and not x.startswith('--'):
                cmd_dd.append('"' + x + '"')
            else:
                cmd_dd.append(x)

        cmd_dd = ' '.join(cmd_dd)
        print(cmd_dd)


        print('%%%%%% {} %%%%%%'.format(self.name))
        print(self.description)
        os.system(cmd_dd)

    
    def remove_dir(self, dir2remove):
        if os.path.exists(dir2remove):
            if sys.platform == 'darwin':
                for fname in glob.glob(os.path.join(dir2remove, '*.hdf5')):
                    os.chflags(fname, not stat.UF_IMMUTABLE)
            shutil.rmtree(dir2remove)

    def run(self):
        self.clean()
        self.execute_cmd()



class Test1(TestObj):
    def __init__(self, *args):
        self.name = 'test_1'
        self.description = 'Generate mask files from Gecko .mjpg files.'
        super().__init__(*args)

        
        self.args = [
        '--video_dir_root',
        self.raw_video_dir,
        '--mask_dir_root',
        self.masked_files_dir,
        '--analysis_type',
        'compress',
        '--pattern_include',
        '*.mjpg',
        '--json_file',
        'filter_worms.json'
        ]

    def clean(self):
        self.remove_dir(self.masked_files_dir)


class Test2(TestObj):
    def __init__(self, *args):
        self.name = 'test_2'
        self.description = 'Generate mask files from .avi files.'
        super().__init__(*args)

        json_file = os.path.join(self.main_dir, 'test2.json')
        self.args = [
        '--video_dir_root',
        self.raw_video_dir,
        '--mask_dir_root',
        self.masked_files_dir,
        '--analysis_type',
        'compress',
        "--json_file",
        json_file,
        '--pattern_include',
        '*.avi'
        ]

    def clean(self):
        self.remove_dir(self.masked_files_dir)

class Test3(TestObj):
    def __init__(self, *args):
        self.name = 'test_3'
        self.description = 'Track from masked video files.'
        super().__init__(*args)

        self.args = [
        '--mask_dir_root',
        self.masked_files_dir,
        '--analysis_type',
        'track',
        "--json_file",
        'filter_worms.json'
        ]

    def clean(self):
        self.remove_dir(self.results_dir)

class Test4(TestObj):
    def __init__(self, *args):
        self.name = 'test_4'
        self.description = 'Calculate features from manually joined trajectories.'
        super().__init__(*args)

        self.feat_manual_file = os.path.join(
        self.main_dir,
        'Results',
        'Capture_Ch1_18062015_140908_feat_manual.hdf5')

        self.args = [
        '--mask_dir_root',
        self.masked_files_dir,
        '--analysis_checkpoints',
        'FEAT_MANUAL_CREATE',
        "--json_file",
        'filter_worms.json'
        ]

    def clean(self):
        if os.path.exists(self.feat_manual_file):
            os.remove(self.feat_manual_file)

class Test5(TestObj):
    def __init__(self, *args):
        self.name = 'test_5'
        self.description = 'Complete analysis from video to calculate features.'
        super().__init__(*args)

        
        self.args = [
        '--mask_dir_root',
        self.masked_files_dir,
        '--analysis_type',
        'track']

    def clean(self):
        self.remove_dir(self.results_dir)

class Test6(TestObj):
    def __init__(self, *args):
        self.name = 'test_6'
        self.description = 'Reformat mask file produced by the rig.'
        super().__init__(*args)

        
        self.args = [
        '--video_dir_root',
        self.raw_video_dir,
        '--mask_dir_root',
        self.masked_files_dir,
        '--analysis_type',
        'compress',
        '--pattern_include',
        '*.raw_hdf5',
        '--json_file',
        'filter_worms.json'
        ]

    def clean(self):
        self.remove_dir(self.masked_files_dir)

class Test7(TestObj):
    def __init__(self, *args):
        self.name = 'test_7'
        self.description = "Schaffer's lab single worm tracker."
        super().__init__(*args)

        extra_cmd = ['STAGE_ALIGMENT', 'CONTOUR_ORIENT', 'FEAT_CREATE', 'WCON_EXPORT']
        self.args = [
        '--video_dir_root',
        self.raw_video_dir,
        '--mask_dir_root',
        self.masked_files_dir,
        '--json_file',
        'single_worm_on_food.json',
        '--pattern_include', 
        '*.avi',
        '--analysis_checkpoints',
        'COMPRESS',
        'COMPRESS_ADD_DATA',
        'VID_SUBSAMPLE',
        'TRAJ_CREATE',
        'TRAJ_JOIN',
        'SKE_INIT',
        'BLOB_FEATS',
        'SKE_CREATE',
        'SKE_FILT',
        'SKE_ORIENT',
        'INT_PROFILE',
        'INT_SKE_ORIENT',
        ]

    def clean(self):
        self.remove_dir(self.masked_files_dir)
        self.remove_dir(self.results_dir)


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

    all_tests_obj = [Test1, Test2, Test3, Test4, Test5, Test6, Test7]

    Test1(examples_dir, script_dir)
    all_tests = [obj(examples_dir, script_dir) for obj in all_tests_obj]

    tests_ind = [x-1 for x in n_tests]
    if tests_ind:
        test_to_exec = [all_tests[x] for x in tests_ind]
    else:
        test_to_exec = all_tests #execute all tests
    
    
    for test in test_to_exec:
        test.run()
