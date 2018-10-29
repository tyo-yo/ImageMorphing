from setuptools import setup, find_packages

def _requirements():
    return [name.rstrip() for name in open('requirements.txt').readlines()]

setup(name='image_morphing',
      version='0.0.1',
      description='image_morphing with python',
      author='Tomoaki Nakamura',
      install_requires=_requirements(),
      packages=find_packages(exclude=('tests', 'docs')),
      url='https://github.com/tyo-yo/ImageMorphing',
)
