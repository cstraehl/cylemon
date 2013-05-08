import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import string 
import distutils

vers = string.split(distutils.__version__, ".")
assert int(vers[0]) >= 2 and int(vers[1]) >= 7 and int(vers[2]) >= 3, """
        \n
        Please make sure you have installed the latest version of setuptools and distutils !!! \n

        update distutils and setuptolls with easy_install :\n
        \n
                    sudo easy_install -U distribute\n
                   """ 

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

try:
    import sys
    from numpy.distutils.misc_util import get_numpy_include_dirs    

    # Check custom args
    include_dirs = [get_numpy_include_dirs()[0]]
    provide_extra_includes = True
    if '--no-extra-includes' in sys.argv:
        i = sys.argv.index('--no-extra-includes')
        sys.argv.pop(i)
        provide_extra_includes = False

    has_openmp = True
    if '--no-openmp' in sys.argv:
        i = sys.argv.index('--no-openmp')
        sys.argv.pop(i)
        has_openmp = False
    
    if sys.platform.startswith("win"):
        for i,d in (include_dirs):
            include_dirs[i] = d.replace('\\', '/')
        libaries = ["lemon"]
        compile_args = ["-O2", "-EHsc"]
        if has_openmp:
            compile_args.append( "-openmp" )
        else:
            compile_args+= ["-D", "CYLEMON_NO_OPENMP"]             
    else:
        if provide_extra_includes:
            include_dirs.append( '/usr/local/include' )
        libaries = ["stdc++", "emon", "gomp"]
        compile_args = ['-O3']
        if has_openmp:
            compile_args.append( "-fopenmp" )
        else:
            compile_args+= ["-D", "CYLEMON_NO_OPENMP"]
    
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
                        language='C++',
                        libraries=libaries,
                        extra_compile_args=compile_args,
                        include_dirs=include_dirs)
                      ]
    )
except Exception as e:
    print """
        
        If the setup.py script fails, this is possibly due to 

            *  missing lemon graph library:
               - please go to  https://lemon.cs.elte.hu/ and download the latest version
               - compile the lemon graph library with the -DBUILD_SHARED_LIBS=1 flag set !!!!!!!!

            * missing openmp libraries
               - install the development packages for openmp for your distribution. 

            * missing include paths for lemon includes or numpy includes
               - add the correct include paths to the "include_dirs" list
                 of the ext_modules section in this file.

    """
    raise e

