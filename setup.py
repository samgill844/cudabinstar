from setuptools import setup, Extension

setup(
    name = 'cudabinstar',
    version = '0.1',
    description = 'GPU accelerated binary star model',
    url = None,
    author = 'Samuel Gill et al',
    author_email = 'samgill844@gmail.com',
    license = 'GNU',
    packages=['cudabinstar','cudabinstar/cudalc','cudabinstar/libso'],

    package_data={'cudabinstar/libso': ['cudalc.so']},
    include_package_data=True,
)