#from distutils.core import setup, find_packages
from setuptools import setup, find_packages

setup(
    name='pyaddgals',
    version='1.0',
#    packages=['PyAddgals',],
    packages=find_packages(),
    scripts=['bin/addgals'],
    package_dir={'PyAddgals' : 'PyAddgals'},
    package_data={'PyAddgals': ['data/filters/*/*', 'data/templates/*']},
    long_description=open('README.md').read(),
    )
