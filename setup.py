from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize, build_ext
import numpy
import os
#try:
#    from Cython.Distutils import build_ext, cythonize
#except ImportError:
#    use_cython = False
#    print("=-=-=-=-=-")
#    print(" Building BANN")
#    print("=-=-=-=-=-")
#else:
#    use_cython = True
#    print("=-=-=-=-=-")
#    print(" Building dev BANN")
#    print("=-=-=-=-=-")
#
##yes Cython dev version
# os.environ["CC"] = "g++"

bann_module_dev = Extension(
        name="bann",
        sources=["bann.pyx", "ann_namespace.cpp"],
        include_dirs=[numpy.get_include(), "cpp_src"],
        language="C++",
        extra_compile_args=["-std=c++11"]
        )
# # No Cython use version
# bann_module = Extension(
#         name = "bann",
#         sources = ["src/bann.cpp", "src/ann_namespace.cpp"],
#         include_dirs = [numpy.get_include(), "src/cpp_src/"],
#         language="c++"
#         )
# #if use_cython:
#    module = cythonize([bann_module_dev])
#else:
#    module = bann_module

setup(
   name = "bann",
   version = "0.0.2",
   author = "Tuyen Pham",
   author_email = "tuyen.pham@ufl.edu",
   install_requires = ['numpy'],
   description = "A Cython wrapper for Bregman Kd-trees",
   packages = find_packages(),
   license = 'MIT',
   python_requires='>=3.11',
   ext_modules = cythonize([bann_module_dev]),
)
