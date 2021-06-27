from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
			Extension("comb_utils_cython",["comb_utils_cython.pyx"],
				libraries=["m"],
				extra_compile_args=["-fopenmp"],
				extra_link_args=['-fopenmp'])
			]

setup(name="comb_utils_cython",cmdclass={"build_ext":build_ext},ext_modules=ext_modules)# cythonize("comb_utils_cython.pyx")

# setup(
# 		ext_modules = cythonize("comb_utils_cython.pyx")
# 	)