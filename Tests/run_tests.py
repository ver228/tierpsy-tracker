
import sys
import os
import shutil
import stat
import glob


def execute_cmd(command):	
	command = ['"' + x + '"' for x in command]
	os.system(' '.join(command))


def remove_dir(dir2remove):
	if os.path.exists(dir2remove):
		for fname in glob.glob(os.path.join(dir2remove, '*.hdf5')):
			os.chflags(fname, not stat.UF_IMMUTABLE)
		shutil.rmtree(dir2remove)

def test1(script_dir, examples_dir):
	print('%%%%%% TEST1 %%%%%%')
	main_dir = os.path.join(examples_dir, 'test_1')
	masked_files_dir = os.path.join(main_dir, 'MaskedVideos')
	raw_video_dir = os.path.join(main_dir, 'RawVideos')
	
	remove_dir(masked_files_dir)
	
	cmd = [sys.executable, os.path.join(script_dir, 'compressMultipleFiles.py'), raw_video_dir, masked_files_dir]
	execute_cmd(cmd)

def test2(script_dir, examples_dir):
	print('%%%%%% TEST2 %%%%%%')
	main_dir = os.path.join(examples_dir, 'test_2')
	masked_files_dir = os.path.join(main_dir, 'MaskedVideos')
	raw_video_dir = os.path.join(main_dir, 'RawVideos')
	json_file = os.path.join(main_dir, 'test2.json')

	remove_dir(masked_files_dir)
	
	cmd = [sys.executable, os.path.join(script_dir, 'compressMultipleFiles.py'), raw_video_dir, masked_files_dir,
	"--json_file", json_file, '--pattern_include', "*.avi"]
	execute_cmd(cmd)

def test3(script_dir, examples_dir):
	print('%%%%%% TEST3 %%%%%%')
	main_dir = os.path.join(examples_dir, 'test_3')
	masked_files_dir = os.path.join(main_dir, 'MaskedVideos')
	results_dir = os.path.join(main_dir, 'Results')
	json_file = os.path.join(main_dir, 'test3.json')

	remove_dir(results_dir)
	cmd = [sys.executable, os.path.join(script_dir, 'trackMultipleFiles.py'), masked_files_dir, 
	'--json_file', json_file]
	execute_cmd(cmd)

def test4(script_dir, examples_dir):
	print('%%%%%% TEST4 %%%%%%')
	main_dir = os.path.join(examples_dir, 'test_4')
	masked_files_dir = os.path.join(main_dir, 'MaskedVideos')
	json_file = os.path.join(examples_dir, 'test_3', 'test3.json')

	feat_manual_file = os.path.join(main_dir, 'Results', 'Capture_Ch1_18062015_140908_feat_manual.hdf5')
	if os.path.exists(feat_manual_file):
		os.remove(feat_manual_file)

	cmd = [sys.executable, os.path.join(script_dir, 'trackMultipleFiles.py'), masked_files_dir, 
	'--json_file', json_file, '--use_manual_join']
	execute_cmd(cmd)

def test5(script_dir, examples_dir):
	print('%%%%%% TEST5 %%%%%%')
	main_dir = os.path.join(examples_dir, 'test_5')
	results_dir = os.path.join(main_dir, 'Results')
	masked_files_dir = main_dir
	
	remove_dir(results_dir)
	cmd = [sys.executable, os.path.join(script_dir, 'trackMultipleFiles.py'), masked_files_dir]
	execute_cmd(cmd)

if __name__ == '__main__':
	examples_dir = os.path.join(os.path.expanduser("~"), 'Google Drive', 'MWTracker', 'Tests')
	script_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', 'cmd_scripts')

	for fun in [test1, test2, test3, test4, test5]:
		fun(script_dir, examples_dir)

	