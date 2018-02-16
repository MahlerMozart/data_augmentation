from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['math', 'tensorflow', 'pillow', 'numpy']
FOUND_PACKAGES = find_packages()
IGNORE_PACKAGES = ['tests']
KEEP_PACKAGES = [i_pack for i_pack in FOUND_PACKAGES if i_pack not in IGNORE_PACKAGES]

setup(name='data_augmentation',
      version='0.1',
      description='data augmentation for various usecases',
      url='https://github.com/berge-brain/data_augmentation',
      author='Jonas Karlsson',
      author_email='jonashtkarlsson@gmail.com',
      license='MIT',
      packages=KEEP_PACKAGES,
      install_requires=REQUIRED_PACKAGES,
      zip_safe=False)
