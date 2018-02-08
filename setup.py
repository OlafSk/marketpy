from setuptools import setup

setup(name='marketpy',
      version='0.1',
      description='Python library for financial modeling',
      url='http://github.com/storborg/funniest',
      author='Olaf Skrabacz',
      author_email='olafsk123@gmail.com',
      license='MIT',
      packages=['marketpy'],
      install_requires=[
          'numpy',
          'sklearn',
          'pandas'
      ],
      zip_safe=False)
