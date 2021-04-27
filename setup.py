#!/usr/bin/env python
from __future__ import absolute_import

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import setup, find_packages

setup(
    name='cluster_gnn',
    version='0.1.0',
    author='Jacan Chaplais',
    license='MIT',
    description='Library for creating jet clustering graph neural networks',
    url="https://github.com/jacanchaplais/cluster_gnn",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 2 - Pre-Alpha',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
    ],
    keywords=[
        'Neural Networks'
        # eg: 'keyword1', 'keyword2', 'keyword3',
        ],
)
