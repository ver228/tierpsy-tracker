#!/usr/bin/env python

from setuptools import setup


exec(open('MWTracker/version.py').read())
setup(name='MWTracker',
      version=__version__,
      description='Multiworm Tracker',
      author='Avelino Javer',
      author_email='avelino.javer@imperial.ac.uk',
      url='https://github.com/ver228/Multiworm_Tracking',
      packages=['MWTracker'],
     )