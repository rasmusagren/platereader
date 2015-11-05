from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='PlateReader',
      version='0.1',
      description='Software for counting and quantifying colonies on nutrient plates',
      url='http://github.com/rasmusagren/platereader',
      author='Rasmus Agren',
      author_email='rasmus@alembic.se',
      license='GNU General Public License v2.0',
      packages=['platereader'],
      include_package_data=True,
      install_requires=[
          'numpy',
          'cv2',
          'matplotlib',
          'pygtk'],
      zip_safe=False)