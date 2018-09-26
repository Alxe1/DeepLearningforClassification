#-*-coding:utf-8-*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([Extension("Squeue",["Squeue.pyx"])])
)

#ext_modules = [Extension("primes",sources=["primes.pyx"])]
#setup(name='primes',ext_modules=cythonize(ext_modules))