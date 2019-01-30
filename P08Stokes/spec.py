import numpy
import scipy.fftpack

def sol_a_x_spec(size, a_x_u):
    n = size
    # Computational cost of lambda is rather small compared to DCT and DST
    lamda = 16.0 * n**2 * (
      numpy.sin(numpy.linspace(0.0, numpy.pi / 2.0, n+1)[1:-1, None])**2
    + numpy.sin(numpy.linspace(0.0, numpy.pi / 2.0, n+1)[None, :-1])**2
    ) 
    a_x_u_hat = scipy.fftpack.dst(a_x_u, type=1, axis=0)
    a_x_u_hat = scipy.fftpack.dct(a_x_u_hat, type=2, axis=1)
    u_hat = a_x_u_hat
    del a_x_u_hat
    u_hat /= lamda
    u_sol = scipy.fftpack.idst(u_hat, type=1, axis=0)
    u_sol = scipy.fftpack.idct(u_sol, type=2, axis=1)
    del u_hat
    return u_sol

def sol_a_y_spec(size, a_y_v):
    n = size
    lamda = 16.0 * n**2 * (
      numpy.sin(numpy.linspace(0.0, numpy.pi / 2.0, n+1)[None, 1:-1])**2
    + numpy.sin(numpy.linspace(0.0, numpy.pi / 2.0, n+1)[:-1, None])**2
    )
    a_y_v_hat = scipy.fftpack.dst(a_y_v, type=1, axis=1)
    a_y_v_hat = scipy.fftpack.dct(a_y_v_hat, type=2, axis=0)
    v_hat = a_y_v_hat
    del a_y_v_hat
    v_hat /= lamda
    v_sol = scipy.fftpack.idst(v_hat, type=1, axis=1)
    v_sol = scipy.fftpack.idct(v_sol, type=2, axis=0)
    del v_hat
    return v_sol