# Multi-grid methods to Stokes equation

According to the requirement, we briefly describe the layouts of codes.

The first part is Python implementations of algorithms.
1. `models.py`: model problems.
2. `mat.py`: construction of matrices and matrix-free operators.
3. `gs.py`: Gauss-Seidel iterations, matrix-free version in Python.
4. `dgs.py`: distributive relaxation of distributive Gauss-Seidel iterations, matrix-free version in Python.
5. `pro.py`: prolongation operators, matrix-free version in Python.
6. `res.py`: restriction operators, matrix-free version in Python.
7. `spec.py`: spectral method (DST and DCT) solvers to Poisson problem in Python.
8. `drivers.py`: driver routines of multi-grid iterations.

The second part is the C counterpart of codes above. All of these codes invoke OpenMP for parallel mechanism.
3. `op/gs.c`: Gauss-Seidel iterations, matrix-free version in C.
4. `op/dgs.c`: distributive relaxation of distributive Gauss-Seidel iterations, matrix-free version in C.
5. `op/pro.c`: prolongation operators, matrix-free version in C.
6. `op/res.c`: restriction operators, matrix-free version in C.
7. `op/spec.c`: spectral method (DST and DCT) solvers to Poisson problem in C, invoking FFTW package.
8. `op/cg.c`: conjugate gradient iterations, matrix-free version in C.
9. `op/mg.c`: Gauss-Seidel multi-grid solvers to Poisson problem, matrix-free version in C.
10. `op/pcg.c`: preconditioned conjugate gradient iterations, matrix-free version in C.
11. `op/op.h`: header file of declarations.
10. `op/wrappers.c`: C to Python wrappers of functions.

The third part is routines calling functions and conduct numerical experiments.
1. `Problem0.py`.
2. `Problem1.py`.
3. `Problem2.py`.
4. `Problem3.py`.
5. `Problem4.py`.
6. `ProblemCorr.py`.
