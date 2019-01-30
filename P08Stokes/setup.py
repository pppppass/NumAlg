from distutils.core import setup, Extension
import numpy
op = Extension(
    name="op",
    sources=[
        "op/gs.c", "op/dgs.c",
        "op/mat.c",
        "op/pro.c", "op/res.c",
        "op/spec.c", "op/cg.c",
        "op/mg.c", "op/pcg.c",
        "op/wrappers.c"
    ],
    include_dirs=[numpy.get_include()],
# Link FFTW before MKL and use -Bsymbolic, see 
# https://groups.google.com/a/continuum.io/forum/#!msg/anaconda/zXT3dRsKK10/DWFCGS31CAAJ
# But we finally drop FFTW, so no special treatment here
    libraries=["fftw3", "fftw3_threads"],#, "mkl_rt"],
    extra_compile_args=["-fopenmp"],
    extra_link_args=["-fopenmp"]#, "-Bsymbolic"]
)

setup(ext_modules=[op])
