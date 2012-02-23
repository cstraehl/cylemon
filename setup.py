import os
from setuptools import setup

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
    package_data={'cylemon' : ["*.py", "lemon/*.pxd", "lemon/*.py", "*.hxx", "*.pyx", "*.pyxbld"]}
)
