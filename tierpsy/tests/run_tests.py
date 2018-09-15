import tierpsy

import sys
import os
import shutil
import stat
import glob
import json
import argparse
import tqdm
import requests
import math
import zipfile
import warnings

DLFT_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
EXAMPLES_LINK="https://imperiallondon-my.sharepoint.com/personal/ajaver_ic_ac_uk/_layouts/15/guestaccess.aspx?guestaccesstoken=ldZ18fLY%2bzlu7XuO9mbKVdyiKoH4naiesqiLXWU4vGQ%3d&docid=0cec4e52f4ccf4d5b8bb3a737020fc12f&rev=1"

def download_files(data_dir):
    # Streaming, so we can iterate over the response.
    r = requests.get(EXAMPLES_LINK, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0)); 
    block_size = 1024
    wrote = 0 

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    tmp_file = os.path.join(data_dir, 'test_data.zip')
    with open(tmp_file, 'wb') as f:
        for data in tqdm.tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='kB', unit_scale=False):
            wrote = wrote  + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong") 

    
    with zipfile.ZipFile(tmp_file, "r") as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(tmp_file)

    #the files would be unzipped into data_dir/data/ I want to move them back to data_dir
    src_dir = os.path.join(data_dir, 'data')
    files = os.listdir(src_dir)
    for f in files:
        shutil.move(os.path.join(src_dir,f), data_dir)
    os.rmdir(src_dir)

class TestObj():
    def __init__(self, examples_dir, base_script):
        self.commands = []
        self.base_script = base_script
        self.main_dir = os.path.join(examples_dir, self.name)
        self.masked_files_dir = os.path.join(self.main_dir, 'MaskedVideos')
        self.raw_video_dir = os.path.join(self.main_dir, 'RawVideos')
        self.results_dir = os.path.join(self.main_dir, 'Results')

    def add_command(self, args, base_script=''):
        if not base_script:
            base_script = self.base_script

        if not '--is_debug' in args:
            args.append('--is_debug')

        command = base_script + args

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
    name = 'GECKO_VIDEOS'
    description = 'Complete analysis from video from Gecko .mjpg files.'
        
    def __init__(self, *args):
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
        '_AEX_RIG.json'
        ]
        self.add_command(args)

class AVI_VIDEOS(TestObj):
    name = 'AVI_VIDEOS'
    description = 'Complete analysis from .avi files.'
    
    def __init__(self, *args):
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
    name = 'MANUAL_FEATS'
    description = 'Calculate features from manually joined trajectories.'
    
    def __init__(self, *args):
        super().__init__(*args)

        args = [
        '--mask_dir_root',
        self.masked_files_dir,
        '--analysis_checkpoints',
        'FEAT_MANUAL_CREATE'
        ]
        self.add_command(args)

    def clean(self):
        fnames = glob.glob(os.path.join(self.main_dir, 'Results', '*_feat_manual.hdf5'))
        for fname in fnames:
            os.remove(fname)


class RIG_HDF5_VIDEOS(TestObj):
    name = 'RIG_HDF5_VIDEOS'
    description = 'Reformat hdf5 file produced by the gecko plugin in the worm rig.'    
    def __init__(self, *args):
        super().__init__(*args)

        args = [
        '--video_dir_root',
        self.raw_video_dir,
        '--mask_dir_root',
        self.masked_files_dir,
        '--pattern_include',
        '*.raw_hdf5',
        '--json_file',
        '_AEX_RIG.json'
        ]
        self.add_command(args)


class WT2(TestObj):
    name = 'WT2'
    description = "Worm Tracker 2.0 (Schafer's lab single worm)."
    def __init__(self, *args):
        super().__init__(*args)

        args = [
        '--video_dir_root',
        self.raw_video_dir,
        '--mask_dir_root',
        self.masked_files_dir,
        '--results_dir_root',
        self.results_dir,
        '--json_file',
        'WT2_clockwise.json',
        '--pattern_include', 
        '*.avi',
        ]
        self.add_command(args)


class WORM_MOTEL(TestObj):
    name = 'WORM_MOTEL'
    description = "Worm motel (background subtraction)."
    def __init__(self, *args):
        super().__init__(*args)

        args = [
        '--video_dir_root',
        self.raw_video_dir,
        '--mask_dir_root',
        self.masked_files_dir,
        '--results_dir_root',
        self.results_dir,
        '--json_file',
        '_AEX_RIG_worm_motel.json',
        '--pattern_include', 
        '*.mjpg'
        ]
        self.add_command(args)


def tierpsy_tests():
    base_script = ['tierpsy_process']
    
    _all_tests_obj = [
                GECKO_VIDEOS, 
                AVI_VIDEOS, 
                MANUAL_FEATS, 
                RIG_HDF5_VIDEOS, 
                WT2, 
                WORM_MOTEL
                ]
    _available_tests = [x.name for x in _all_tests_obj]
    _available_tests_str = ' '.join(_available_tests)
    test_dict = dict(zip(_available_tests, _all_tests_obj))

    help_test = 'Name of the tests to be executed. If not values are given all tests will be executed. The available tests are: {}'.format(_available_tests_str)

    parser = argparse.ArgumentParser(description='Tierpsy Tracker tests.')
    parser.add_argument('tests', 
                        nargs='*',
                        help=help_test)
    
    parser.add_argument('--download_files',
                        action = 'store_true',
                        help = 'Flag to indicate if the test files are going to be downloaded.')
    
    parser.add_argument('--data_dir',  
                        default = DLFT_DATA_DIR,
                        help='Directory where the test files are located or where they will be downloaded.'
                        )
    

    args = parser.parse_args()
    data_dir = args.data_dir


    if args.download_files:
        download_files(data_dir)

    if not os.path.exists(data_dir):
        print('The given --data_dir "{}" does not exists. Select the directory with the valid files or use --download_files.'.format(data_dir))
        return

    tests_given = args.tests
    
    test2run = []
    for tt in tests_given:
        if tt in _available_tests:
            test2run.append(tt)
        else:
            warnings.warn('Test "{}" is not a valid name, and it will be skiped. The valid tests are: {}'.format(tt, _available_tests_str))
    
    if not tests_given:
        print("No tests given. Please specify some valid tests {}.".format(_available_tests_str))
        return

    for test_name in test2run:
        test = test_dict[test_name](args.data_dir, base_script)
        test.run()
