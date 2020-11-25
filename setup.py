import setuptools

with open('README.md', 'r') as fh:
  long_description = fh.read()

setuptools.setup(
  name = 'mymi',
  packages = setuptools.find_packages(),
  version = '0.0.1',
  description = 'Medical imaging code',
  long_description = long_description,
  author = 'Brett Clark',
  author_email = 'clarkbab@gmail.com',
  url = 'https://github.com/clarkbab/medical-imaging',
  keywords = ['medical imaging'],
  classifiers = []
)