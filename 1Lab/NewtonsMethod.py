def newtons_method(x0):
    for i in range(0, 1000):
        xn = x0 - (x0 * (x0 ** 2 - 4) / 2) / (3 * x0 ** 2 / 2 - 2)
        if abs(xn - x0) < 0.0001:
            break
        x0 = xn

    print("Number of iterations: %d" % i)
    print("xn: %f" % xn)


newtons_method("((x0**2 - 4)**2)/8", 5)
