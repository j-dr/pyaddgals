from distutils.core import setup

setup(
    name='pyaddgals',
    version='1.0',
    packages=['PyAddgals',],
    scripts=['bin/addgals'],
    package_dir={'PyAddgals' : 'PyAddgals'},
    package_data={'PyAddgals': ['data/filters/*/*', 'data/templates/*']},
    long_description=open('README.md').read(),
    )
