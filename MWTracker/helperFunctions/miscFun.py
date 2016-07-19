import shutil
import sys
import os
from MWTracker import AUX_FILES_DIR

WLAB = {'U':0, 'WORM':1, 'WORMS':2, 'BAD':3, 'GOOD_SKE':4}

def print_flush(msg):
	print(msg)
	sys.stdout.flush()

#get the correct path for ffmpeg. First we look in the auxFiles directory, otherwise we look in the system path.
def get_local_or_sys_path(file_name):
    file_source = os.path.join(AUX_FILES_DIR, file_name)
    if not os.path.exists(file_source):
        file_source = shutil.which(file_name)
    
    if not file_source:
        raise FileNotFoundError('command not found: %s' % file_name)
    return file_source