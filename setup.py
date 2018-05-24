#!/usr/bin/env python

from os.path import exists
from setuptools import setup


setup(name='functoolsex',
      version='0.0.6',
      description='return ex(pyrsistent, fn, pytoolz, ...)',
      url='https://github.com/aymazon/functools-ex',
      author='Tony L. Fan',
      author_email='aymazon@gmail.com',
      license='MIT',
      keywords='fn pyrsistent pytoolz functional utility itertools functools',
      packages=['functoolsex'],
      package_data={'functoolsex': ['tests/*.py']},
      long_description=(open('README.rst').read() if exists('README.rst')
                        else ''),
      install_requires=(open('requirements.txt').readlines() if exists('requirements.txt')
                        else []),
      zip_safe=False,
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python",
          "Programming Language :: Python :: 2.6",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: Implementation :: CPython",
          "Programming Language :: Python :: Implementation :: PyPy"])
