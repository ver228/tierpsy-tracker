# # -*- coding: utf-8 -*-
__version__ = '0.1'

try:
    import os
    import subprocess

    cwd = os.path.dirname(os.path.abspath(__file__))
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
    __version__ += '+' + sha[:7]
except Exception:
    pass