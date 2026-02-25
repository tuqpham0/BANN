from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize, build_ext
import numpy
from os import path

# here = path.abspath(path.dirname(__file__))
# with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#     long_description = f.read()

bann_module = Extension(
   name="bann",
   sources=["bann.pyx", "ann_namespace.cpp"],
   include_dirs=[numpy.get_include(), "src/", "cpp_src/"],
   language="c++"
)

setup(
   name = "bann",
   version = "0.0.1",
   author = "Tuyen Pham",
   author_email = "tuyen.pham@ufl.edu",
   setup_requires = [
                  'setuptools>=18.0',
                  'cython'
                  ],
   install_requires = ['numpy'],
   description = "A Cython wrapper for Bregman ANN searches",
   packages = find_packages(),
   license = 'MIT',
   python_requires='>=3.11',
   ext_modules = cythonize([bann_module])
)
