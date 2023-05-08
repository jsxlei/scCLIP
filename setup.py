#!/usr/bin/env python
"""
# Author: Lei Xiong
# Created Time : Sun 17 Nov 2021 03:37:47 PM CST

# File Name: setup.py
# Description:

"""
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='scclip',
    version='0.0.1',
    packages=find_packages(),
    description='',
    test_suite='tests',

    author='Lei Xiong',
    author_email='jsxlei@gmail.com',
    url='https://github.com/jsxlei/scclip',
    scripts=[],
    install_requires=requirements,
    python_requires='>3.6.0',
    license='MIT',

    classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.7',
      'Operating System :: MacOS :: MacOS X',
      'Operating System :: Microsoft :: Windows',
      'Operating System :: POSIX :: Linux',
      'Topic :: Scientific/Engineering :: Bio-Informatics',
     ],
     )


