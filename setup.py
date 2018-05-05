from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='mlkit',
   version='0.1',
   description='Toolkit for machine learning.',
   license="MIT",
   long_description=long_description,
   author='Hoiy',
   author_email='hoiy927@gmail.com',
   url="",
   packages=['mlkit'],
   install_requires=[]
)
