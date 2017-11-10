from setuptools import setup

setup(
    name='vulcanai',
    version='0.1',
    description='A high-level framework built on top of Theano and Lasagne'
                ' using added functionality from Scikit-learn to provide '
                'all of the tools needed for visualizing high-dimensional'
                ' data, modular neural networks, and model evaluation',
    url='https://github.com/rfratila/Vulcan',
    install_requires=['numpy>=1.12.0',
                      'scipy>=0.17.1',
                      'matplotlib>=1.5.3',
                      'scikit-learn>=0.18',
                      'jsonschema>=2.6.0',
                      'theano>=0.9.0',
                      'lasagne>=0.2.dev1'],
    packages=['vulcanai']
)
