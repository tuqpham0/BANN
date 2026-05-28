from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
import os

bann_module_dev = Extension(
        name="bann",
        sources=["bann.pyx", "ann_namespace.cpp"],
        include_dirs=[numpy.get_include(), "cpp_src"],
        language="C++",
        extra_compile_args=["-std=gnu++14"]
        )

setup(
   name = "bann",
   version = "0.0.3",
   author = "Tuyen Pham",
   author_email = "tuyen.pham@ufl.edu",
   install_requires = ['numpy'],
   description = "A Cython wrapper for Bregman Kd-trees",
   packages = find_packages(),
   license = 'MIT',
   python_requires='>=3.11',
   ext_modules = cythonize([bann_module_dev]),
)
