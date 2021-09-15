'''(x^2-4)^2/8'''


def bisection_method(func, l, r, eps):
    def f(x):
        return eval(func)

    xm = (l + r) / 2
    L = r - l

    for i in range(0, 100):
        x1 = l + L / 4
        x2 = r - L / 4

        if f(x1) < f(xm):
            r = xm
            xm = x1
            L = r - l
            if L < eps:
                break
        elif f(x2) < f(xm):
            r = xm
            xm = x2
            L = r - l
            if L < eps:
                break
        else:
            l = x1
            r = x2
            L = r - l
            if L < eps:
                break

    print("Number of iterations: %d" % i)
    find_min(f(x1), f(x2), f(xm), x1, x2, xm)


def find_min(a, b, c, x1, x2, xm):
    if a < b and a < c:
        print("f(x1) =", a)
        print("x1 =", x1)
    if b < a and b < c:
        print("f(x2) =", b)
        print("x2 =", x2)
    if c < a and c < b:
        print("f(xm) =", c)
        print("xm =", xm)


bisection_method("((x ** 2 - 4) ** 2 /9)-1", 0, 10, 0.0001)
