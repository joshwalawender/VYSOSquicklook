# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.

import sys
import os
import glob

from setuptools import setup

# Get some values from the setup.cfg
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser

conf = ConfigParser()
conf.read(['setup.cfg'])
#metadata = dict(conf.items('metadata'))


# MODIFY THE NAME OF THE PACKAGE to be the one chosen
NAME = 'vysosdrp'
VERSION = '1.0dev'
RELEASE = 'dev' not in VERSION

scripts = []

entry_points = {
    'console_scripts': [
        "analyzeone = vysosdrp.script:analyze_one",
        "watchdirectory = vysosdrp.script:watch_directory",
        "make_nightly_plot = vysosdrp.primitives.make_nightly_plot:main",
        "qlcd = vysosdrp.script:change_directory",
    ]
}


# modify the list of packages, to make sure that your package is defined correctly
setup(name=NAME,
      provides=NAME,
      version=VERSION,
      license='BSD3',
      description='VYSOS Quick Look DRP.',
      long_description=open('README.txt').read(),
      author='Josh Walawender',
      author_email='jmwalawender@gmail.com',
      packages=['vysosdrp',],
      scripts=scripts,
      entry_points=entry_points,
      )
