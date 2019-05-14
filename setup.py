from setuptools import setup, find_packages

setup(name='flerken',
      version='0.2.4',
      description='PyTorch extension with simplified trainer. ',
      url='https://github.com/JuanFMontesinos/flerken',
      author='Juan Montesinos',
      author_email='juanfelipe.montesinos@upf.edu',
      packages=find_packages(),
      install_requires=['tqdm', 'numpy', 'sklearn'],
      classifiers=[
          "Programming Language :: Python :: 3", ],
      zip_safe=False)
