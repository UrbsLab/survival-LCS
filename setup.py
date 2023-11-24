from setuptools import setup

with open("README.md","r") as fh:
  ld = fh.read()

setup(
  name = 'survival-ExSTraCS',
  packages = ['survival-ExSTraCS'],
  version = '1.0.10',
  license='License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
  description = 'Survival Learning Classifier System Discovery and Visualization Environment',
  long_description_content_type="text/markdown",
  author = 'Alexa A. Woodward, Ryan J. Urbanowicz',
  author_email = 'alexa.a.woodward@gmail.com,ryanurb@upenn.edu',
  url = 'https://github.com/alexa-woodward/survival-ExSTraCS',
  keywords = ['machine learning','data analysis','data science','learning classifier systems','dive'],
  install_requires=['numpy','pandas','scikit-learn','seaborn','skrebate','scikit-ExSTraCS','networkx','matplotlib','scipy','sklearn','pygame','fastcluster'],
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Information Technology',
    'Intended Audience :: Science/Research',
    'Topic :: Utilities',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'
  ],
  long_description=ld
)
