from setuptools import setup

with open("README.md","r") as fh:
  ld = fh.read()

setup(
  name = 'survival-LCS',
  packages = ['survival-LCS'],
  version = '1.0.1',
  license='License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
  description = 'Survival Learning Classifier System Discovery and Visualization Environment',
  long_description_content_type="text/markdown",
  author = 'Alexa A. Woodward, Harsh Bandhey, Ryan J. Urbanowicz',
  author_email = 'harsh.bandhey@cshs.org',
  url = 'https://github.com/Urbslab/survival-LCS',
  keywords = ['machine learning','data analysis','data science','learning classifier systems','dive'],
  install_requires=[
    'numpy==1.21.2',
    'pandas',
    'matplotlib',
    'scikit-learn==1.22.2',
    'skrebate',
    'fastcluster',
    'seaborn',
    'networkx',
    'pygame',
    'pytest-shutil',
    'eli5',
    'scikit-survival==0.21.0',
  ]',
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Information Technology',
    'Intended Audience :: Science/Research',
    'Topic :: Utilities',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 3',
  ],
  long_description=ld
)
