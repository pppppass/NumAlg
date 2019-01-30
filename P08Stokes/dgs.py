def iter_dgs_incomp(size, u, v, p, c_i):
    n = size
    h = 1.0 / n
    dq = (c_i[0, 0] + (u[0, 0] + v[0, 0]) * h) / 2.0 #/ h**2
    u[0, 0] -= dq / h #* h
    v[0, 0] -= dq / h #* h
    p[0, 0] -= 4.0 * dq / h**2 #* h**2
    p[1, 0] += dq / h**2 #* h**2
    p[0, 1] += dq / h**2 #* h**2
    for i in range(1, n-1):
        dq = (c_i[0, i] + (u[0, i] + v[0, i] - v[0, i-1]) * h) / 3.0 #/ h**2
        u[0, i] -= dq / h #* h
        v[0, i] -= dq / h #* h
        v[0, i-1] += dq / h #* h
        p[0, i] -= 4.0 * dq / h**2 #* h**2
        p[1, i] += dq / h**2 #* h**2
        p[0, i-1] += dq / h**2 #* h**2
        p[0, i+1] += dq / h**2 #* h**2
    dq = (c_i[0, n-1] + (u[0, n-1] - v[0, n-2]) * h) / 2.0
    u[0, n-1] -= dq / h
    v[0, n-2] += dq / h
    p[0, n-1] -= 4.0 * dq / h**2
    p[1, n-1] += dq / h**2
    p[0, n-2] += dq / h**2
    for j in range(1, n-1):
        dq = (c_i[j, 0] + (u[j, 0] - u[j-1, 0] + v[j, 0]) * h) / 3.0 #/ h**2
        u[j, 0] -= dq / h #* h
        u[j-1, 0] += dq / h
        v[j, 0] -= dq / h #* h
        p[j, 0] -= 4.0 * dq / h**2 #* h**2
        p[j-1, 0] += dq / h**2
        p[j+1, 0] += dq / h**2 #* h**2
        p[j, 1] += dq / h**2 #* h**2
        for i in range(1, n-1):
            dq = (c_i[j, i] + (u[j, i] - u[j-1, i] + v[j, i] - v[j, i-1]) * h) / 4.0 #/ h**2
            u[j, i] -= dq / h #* h
            u[j-1, i] += dq / h
            v[j, i] -= dq / h #* h
            v[j, i-1] += dq / h #* h
            p[j, i] -= 4.0 * dq / h**2 #* h**2
            p[j-1, i] += dq / h**2
            p[j+1, i] += dq / h**2 #* h**2
            p[j, i-1] += dq / h**2 #* h**2
            p[j, i+1] += dq / h**2 #* h**2
        dq = (c_i[j, n-1] + (u[j, n-1] - u[j-1, n-1] - v[j, n-2]) * h) / 3.0
        u[j, n-1] -= dq / h
        u[j-1, n-1] += dq / h
        v[j, n-2] += dq / h
        p[j, n-1] -= 4.0 * dq / h**2
        p[j-1, n-1] += dq / h**2
        p[j+1, n-1] += dq / h**2
        p[j, n-2] += dq / h**2
    dq = (c_i[n-1, 0] + (-u[n-2, 0] + v[n-1, 0]) * h) / 2.0 #/ h**2
    u[n-2, 0] += dq / h #* h
    v[n-1, 0] -= dq / h #* h
    p[n-1, 0] -= 4.0 * dq / h**2 #* h**2
    p[n-2, 0] += dq / h**2 #* h**2
    p[n-1, 1] += dq / h**2 #* h**2
    for i in range(1, n-1):
        dq = (c_i[n-1, i] + (-u[n-2, i] + v[n-1, i] - v[n-1, i-1]) * h) / 3.0 #/ h**2
        u[n-2, i] += dq / h #* h
        v[n-1, i] -= dq / h #* h
        v[n-1, i-1] += dq / h #* h
        p[n-1, i] -= 4.0 * dq / h**2 #* h**2
        p[n-2, i] += dq / h**2 #* h**2
        p[n-1, i-1] += dq / h**2 #* h**2
        p[n-1, i+1] += dq / h**2 #* h**2
    dq = (c_i[n-1, n-1] + (-u[n-2, n-1] - v[n-1, n-2]) * h) / 2.0
    u[n-2, n-1] += dq / h
    v[n-1, n-2] += dq / h
    p[n-1, n-1] -= 4.0 * dq / h**2
    p[n-2, n-1] += dq / h**2
    p[n-1, n-2] += dq / h**2
    return u, v, p