def iter_gs_u(size, u, p, c_x):
    n = size
    h = 1.0 / n
    u[0, 0] = (c_x[0, 0] - (p[1, 0] - p[0, 0]) * h + u[1, 0] + u[0, 1]) / 3.0
    for i in range(1, n-1):
        u[0, i] = (c_x[0, i] - (p[1, i] - p[0, i]) * h + u[1, i] + u[0, i-1] + u[0, i+1]) / 4.0
    u[0, n-1] = (c_x[0, n-1] - (p[1, n-1] - p[0, n-1]) * h + u[1, n-1] + u[0, n-2]) / 3.0
    for j in range(1, n-2):
        u[j, 0] = (c_x[j, 0] - (p[j+1, 0] - p[j, 0]) * h + u[j-1, 0] + u[j+1, 0] + u[j, 1]) / 3.0
        for i in range(1, n-1):
            u[j, i] = (c_x[j, i] - (p[j+1, i] - p[j, i]) * h + u[j-1, i] + u[j+1, i] + u[j, i-1] + u[j, i+1]) / 4.0
        u[j, n-1] = (c_x[j, n-1] - (p[j+1, n-1] - p[j, n-1]) * h + u[j-1, n-1] + u[j+1, n-1] + u[j, n-2]) / 3.0
    u[n-2, 0] = (c_x[n-2, 0] - (p[n-1, 0] - p[n-2, 0]) * h + u[n-3, 0] + u[n-2, 1]) / 3.0
    for i in range(1, n-1):
        u[n-2, i] = (c_x[n-2, i] - (p[n-1, i] - p[n-2, i]) * h + u[n-3, i] + u[n-2, i-1] + u[n-2, i+1]) / 4.0
    u[n-2, n-1] = (c_x[n-2, n-1] - (p[n-1, n-1] - p[n-2, n-1]) * h + u[n-3, n-1] + u[n-2, n-2]) / 3.0
    return u


def iter_gs_v(size, v, p, c_y):
    n = size
    h = 1.0 / n
    v[0, 0] = (c_y[0, 0] - (p[0, 1] - p[0, 0]) * h + v[1, 0] + v[0, 1]) / 3.0
    for i in range(1, n-2):
        v[0, i] = (c_y[0, i] - (p[0, i+1] - p[0, i]) * h + v[1, i] + v[0, i-1] + v[0, i+1]) / 3.0
    v[0, n-2] = (c_y[0, n-2] - (p[0, n-1] - p[0, n-2]) * h + v[1, n-2] + v[0, n-3]) / 3.0
    for j in range(1, n-1):
        v[j, 0] = (c_y[j, 0] - (p[j, 1] - p[j, 0]) * h + v[j-1, 0] + v[j+1, 0] + v[j, 1]) / 4.0
        for i in range(1, n-2):
            v[j, i] = (c_y[j, i] - (p[j, i+1] - p[j, i]) * h + v[j-1, i] + v[j+1, i] + v[j, i-1] + v[j, i+1]) / 4.0
        v[j, n-2] = (c_y[j, n-2] - (p[j, n-1] - p[j, n-2]) * h + v[j-1, n-2] + v[j+1, n-2] + v[j, n-3]) / 4.0
    v[n-1, 0] = (c_y[n-1, 0] - (p[n-1, 1] - p[n-1, 0]) * h + v[n-2, 0] + v[n-1, 1]) / 3.0
    for i in range(1, n-2):
        v[n-1, i] = (c_y[n-1, i] - (p[n-1, i+1] - p[n-1, i]) * h + v[n-2, i] + v[n-1, i-1] + v[n-1, i+1]) / 3.0
    v[n-1, n-2] = (c_y[n-1, n-2] - (p[n-1, n-1] - p[n-1, n-2]) * h + v[n-2, n-2] + v[n-1, n-3]) / 3.0
    return v
