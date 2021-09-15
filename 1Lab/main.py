import numpy as np
from matplotlib import pyplot as plt


def bisection_method(func, l, r, eps):
    def f(x):
        return eval(func)

    xm = (l + r) / 2
    L = r - l

    for i in range(1, 100):
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

    print("Bisection method")
    print("Number of iterations: %d" % i)
    find_min(f(x1), f(x2), f(xm), x1, x2, xm)
    print("***********************")


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


def golden_section(func, l, r, eps, t):
    def f(x):
        return eval(func)

    L = r - l
    x1 = r - t * L
    x2 = l + t * L
    for i in range(1, 100):
        if f(x2) < f(x1):
            l = x1
            L = r - l
            x1 = x2
            x2 = l + t * L
        else:
            r = x2
            L = r - l
            x2 = x1
            x1 = r - t * L

        if L < eps:
            break

    print("Golden section method")
    print("Number of iteratioons %d" % i)
    findMin(f(x1), f(x2), x1, x2)
    print("***********************")


def findMin(a, b, x1, x2):
    if a < b:
        print("f(x1) =", a)
        print("x1 =", x1)
    else:
        print("f(x2) =", b)
        print("x2 =", x2)


def newtons_method(x0):
    for i in range(1, 1000):
        xn = x0 - (x0 * (x0 ** 2 - 4) / 2) / (3 * x0 ** 2 / 2 - 2)
        if abs(xn - x0) < 0.0001:
            break
        x0 = xn

    print("Newton's method")
    print("Number of iterations: %d" % i)
    print("xn: %f" % xn)
    print("**********************")


def createGraph(func):
    x = np.linspace(-5, 5, 100)

    y = eval(func)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.plot(x, y, 'r')

    plt.show()


def main():
    function = "((x ** 2 - 4) ** 2 /9)-1"
    bisection_method(function, 0, 10, 0.0001)
    golden_section(function, 0, 10, 0.0001, 0.61803)
    newtons_method(5)
    createGraph(function)


if __name__ == "__main__":
    main()
