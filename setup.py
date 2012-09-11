import os
from setuptools import setup
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "cylemon",
    version = "0.0.1",
    author = "Christoph Straehle",
    author_email = "christoph.straehle@iwr.uni-heidelberg.de",
    description = ("very partial cython bindings for the lemon graph library"),
    license = "BSD",
    keywords = "cython python lemon bindings",
    url = "",
    packages=['cylemon'],
    long_description=read('README'),
    package_dir={'cylemon': "cylemon"},
    package_data={'cylemon' : ["*.py", "lemon/*.pxd", "lemon/*.py", "*.hxx", "*.pyx", "*.pyxbld", "*.so", "*.pxd"]},

    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension(name="cylemon.segmentation",
                    sources=["cylemon/segmentation.pyx"],
                    libraries = ["stdc++", "emon", "gomp"],
                    language='C++',
                    extra_compile_args=['-O3', '-fopenmp'])
                  ]
)
