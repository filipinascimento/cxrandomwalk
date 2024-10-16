#!/usr/bin/env python3
from setuptools import setup, Extension, Command
import os.path
import os
import platform
import setuptools
import sys
import glob
print("Building on:", sys.version)

with open("requirements.txt", "r", encoding="utf8") as fh:
    requirements = fh.readlines()

enableParallelism = True

extraOptions = []
extraLibraryPaths = []
extraIncludesPaths = []
extraLinkOptions=[]
compilerOptions = [
                # "-g",
                "-std=c11",
                # "-m64",
                # "-g3",
                # "-O0",
                "-Wall",
                "-Wno-unused-function",
                "-Wno-deprecated-declarations",
                "-Wno-sign-compare",
                "-Wno-strict-prototypes",
                "-Wno-unused-variable",
                "-O3",
                # "-fvisibility=hidden",
                "-funroll-loops",
                "-fstrict-aliasing"
            ]
if(platform.system()=="Darwin"):
    extraOptions = ["-D OSX"]
    if(enableParallelism):
        extraOptions += ["-DCV_USE_LIBDISPATCH=1"]
elif(platform.system()=="Windows"):
    # extraOptions += ["-D WIN32 -lpthread"]
    extraOptions += ["/D WIN32"]
    extraOptions += ["/D __WIN32__"]
    compilerOptions = [
                "/std:c11",
                "/Wall",
                "/O2",
                # "-funroll-loops",
                # "-fstrict-aliasing"
            ]
    if(enableParallelism):
        # extraOptions += ["-DCV_USE_OPENMP=1","-fopenmp"]
        # extraLinkOptions += ["-lgomp"]
        # extraLinkOptions+=["-lgomp"]
        extraOptions+=["/D CV_USE_OPENMP=1"]
        extraOptions+=["/openmp"]
    
    if("VCPKG_INSTALLATION_ROOT" in os.environ):
        extraIncludesPaths += [os.path.join(os.environ["VCPKG_INSTALLATION_ROOT"], "installed", "x64-windows-static","include")]
        extraLibraryPaths += [os.path.join(os.environ["VCPKG_INSTALLATION_ROOT"], "installed", "x64-windows-static","lib")]

elif(platform.system()=="Linux"):
    extraOptions = ["-D Linux","-D_GNU_SOURCE=1"]
    if(enableParallelism):
        extraOptions += ["-DCV_USE_OPENMP=1","-fopenmp"]
        extraLinkOptions+=["-lgomp"]
else:
    if(enableParallelism):
        extraOptions += ["-DCV_USE_OPENMP=1","-fopenmp"]
        extraLinkOptions+=["-lgomp"]

# WORKAROUND: https://stackoverflow.com/questions/54117786/add-numpy-get-include-argument-to-setuptools-without-preinstalled-numpy
class get_numpy_include(object):
    def __str__(self):
        import numpy
        return numpy.get_include()

with open("README.md", "r") as fh:
    long_description = fh.read()

building_on_windows = platform.system() == "Windows"

print("!!!!!Building on",platform.system(),"!!!!!!")


packageName = "cxrandomwalk"
packageDirectory = "cxrandomwalk"
extensionPackageName = "cxrandomwalk_core"


with open(os.path.join(packageDirectory, "Python", "PyCXVersion.h"), "rt") as fd:
    version = fd.readline().strip().split(" ")[-1]

print("Compiling version %s"%version)

print([req for req in requirements if req[:2] != "# "])
setup(
    name=packageName,
    version=version,
    author="Filipi N. Silva",
    author_email="filipinascimento@gmail.com",
    # compiler = "mingw32" if building_on_windows else None,
    install_requires=[req for req in requirements if req[:2] != "# "],
    setup_requires=["wheel","numpy"],
    description="Library to perform random walks on complex networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/filipinascimento/cxrandomwalk",
    packages=setuptools.find_packages(),
    classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX :: Linux",
            "Development Status :: 3 - Alpha",
            "Programming Language :: C",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Intended Audience :: Science/Research"
    ],
    python_requires='>=3.9',
    ext_modules = [
        Extension(
            extensionPackageName,
            sources=[
				os.path.join(packageName,"Source", "CVSimpleQueue.c"),
				os.path.join(packageName,"Source", "CVSet.c"),
				os.path.join(packageName,"Source", "CVNetwork.c"),
				os.path.join(packageName,"Source", "CVDictionary.c"),
				os.path.join(packageName,"Source", "CVDistribution.c"),
				os.path.join(packageName,"Source", "fib.c"),
				os.path.join(packageName,"Source", "CVNetworkCentrality.c"),
				os.path.join(packageName,"Python", "PyCXRandomWalk.c"),
            ],
            include_dirs=[
                os.path.join(packageDirectory,"Source"),
                os.path.join(packageDirectory,"Python"),
                get_numpy_include()
            ]+extraIncludesPaths,
            extra_compile_args=compilerOptions+extraOptions,
            extra_link_args=extraLinkOptions,
            # library_dirs=extraLibraryPaths,
            # libraries = ["getopt"] if building_on_windows else [],
        ),
    ]
)

        
