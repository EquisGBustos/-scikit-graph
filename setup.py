from setuptools import setup
from setuptools import find_packages

setup(name='prueba',
      version='1.0',
      description='scikit learn extention for graphs',
      author='me',
      author_email='f@gmail.com',
      url='https://github.com/EquisGBustos/-scikit-graph',
      download_url='https://github.com/EquisGBustos/-scikit-graph',
      license='MIT',
      install_requires=['numpy',
                        'networkx',
                        'scipy',
                        'pandas'
                        ],
      package_data={'all_data': ['README.md']},
packages=find_packages())
