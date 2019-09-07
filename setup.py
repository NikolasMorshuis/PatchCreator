# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='patchcreator',
    version='0.0.1',
    description='Package that helps to predict large files using patches',
    long_description=readme,
    author='Jan Nikolas Morshuis',
    author_email='jnmorshuis@gmail.com',
    url='https://github.com/nikolasmorshuis/patchcreator',
    license=license,
    packages=find_packages(exclude=('docs'))
)

