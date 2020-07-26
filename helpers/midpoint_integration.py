def midpoint2D(f, f_limx, f_limy, nx=10, ny=10, **kwargs):
    """calculates a double integral using the midpoint rule"""
    I = 0
    # start with outside (y) limits of integration
    c, d = f_limy(**kwargs)
    hy = (d - c) / float(ny)
    for j in range(ny):
        yj = c + hy / 2 + j * hy
        # for each j, calculate inside limits of integration
        a, b = f_limx(yj, **kwargs)
        hx = (b - a) / float(nx)
        for i in range(nx):
            xi = a + hx / 2 + i * hx
            I += hx * hy * f(xi, yj, **kwargs)
    return I
