from setuptools import setup

setup(name='flerken',
      version='0.1',
      description='PyTorch extension with simplified trainer. ',
      url='https://github.com/JuanFMontesinos/flerken',
      author='Juan Montesinos',
      author_email='juanfelipe.montesinos@upf.edu',
      packages=['flerken'],
      install_requires = ['tqdm','tensorboardX','numpy','logging'],
      zip_safe=False)
