# # -*- coding: utf-8 -*-
__version__ = '0.1'

try:
    import os
    import subprocess

    cwd = os.path.dirname(os.path.abspath(__file__))
    command = ['git', 'rev-parse', 'HEAD']    
    sha = subprocess.check_output(command, cwd=cwd, stderr = subprocess.DEVNULL).decode('ascii').strip()
    __version__ += '+' + sha[:7]
except Exception:
    pass