from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need
# fine tuning.
buildOptions = dict(packages = [], excludes = ['PyQt4', 'PyQt4.QtCore'])

base = 'Console'

executables = [
    Executable('dum.py', base=base)
]

setup(name='dum',
      version = '1.0',
      description = 'test',
      options = dict(build_exe = buildOptions),
      executables = executables)
