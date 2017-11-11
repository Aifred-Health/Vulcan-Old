"""Setup script for uploading package to PyPI servers."""
from setuptools import setup

setup(
    name='vulcanai',
    version='0.1rc',
    description='A high-level framework built on top of Theano and Lasagne'
                ' using added functionality from Scikit-learn to provide '
                'all of the tools needed for visualizing high-dimensional'
                ' data, modular neural networks, and model evaluation',
    author='Robert Fratila',
    author_email='robertfratila10@gmail.com',
    url='https://github.com/rfratila/Vulcan',
    install_requires=['numpy>=1.12.0',
                      'scipy>=0.17.1',
                      'matplotlib>=1.5.3',
                      'scikit-learn>=0.18',
                      'jsonschema>=2.6.0',
                      'theano>=0.9.0',
                      'lasagne>=0.2.dev1'],
    packages=['vulcanai'],
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Education',
                 'Topic :: Software Development :: Build Tools',
                 'Programming Language :: Python :: 2.7',
                 'Operating System :: Unix',
                 'Operating System :: POSIX :: Linux',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
    keywords='deep learning machine learning development'
)
