from setuptools import setup, find_packages
import re
VERSIONFILE="flerken/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(name='flerken-nightly',
      version=verstr,
      description='PyTorch extension with simplified trainer. ',
      url='https://github.com/JuanFMontesinos/flerken',
      author='Juan Montesinos',
      author_email='juanfelipe.montesinos@upf.edu',
      packages=find_packages(),
      install_requires=['tqdm', 'numpy', 'sklearn'],
      classifiers=[
          "Programming Language :: Python :: 3", ],
      zip_safe=False)
