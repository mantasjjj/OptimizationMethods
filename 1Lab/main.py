import random

import numpy as np
from matplotlib import pyplot as plt

bisec_method_points = []
newton_method_points = []
golden_sec_method_points = []


def bisection_method(func, l, r, eps):
    counter = 0

    def f(x):
        return eval(func)

    xm = (l + r) / 2
    L = r - l
    fxm = f(xm)
    counter += 1

    for i in range(1, 100):
        x1 = l + L / 4
        x2 = r - L / 4

        fx1 = f(x1)
        counter += 1
        if fx1 >= fxm:
            fx2 = f(x2)
            counter += 1

        if fx1 < fxm:
            r = xm
            xm = x1
            fxm = fx1
            L = r - l
            if L < eps:
                break
        elif fx2 < fxm:
            l = xm
            xm = x2
            fxm = fx2
            L = r - l
            if L < eps:
                break
        else:
            l = x1
            r = x2
            L = r - l
            if L < eps:
                break
        bisec_method_points.append([l, r])
    print("Bisection method")
    print("Number of iterations: %d" % i)
    find_min(f(x1), f(x2), f(xm), x1, x2, xm)
    print("Number of functions calculated: %d" % counter)
    print("***********************")


def find_min(a, b, c, x1, x2, xm):
    if a <= b and a <= c:
        print("f(x1) =", a)
        print("x1 =", x1)
    if b <= a and b <= c:
        print("f(x2) =", b)
        print("x2 =", x2)
    if c <= a and c <= b:
        print("f(xm) =", c)
        print("xm =", xm)


def golden_section(func, l, r, eps, t):
    counter = 0
    def f(x):
        return eval(func)

    L = r - l
    x1 = r - t * L
    x2 = l + t * L
    fx1 = f(x1)
    fx2 = f(x2)
    counter += 2
    for i in range(1, 100):
        if fx2 < fx1:
            l = x1
            L = r - l
            x1 = x2
            fx1 = fx2
            x2 = l + t * L
            fx2 = f(x2)
            counter += 1
        else:
            r = x2
            L = r - l
            x2 = x1
            fx2 = fx1
            x1 = r - t * L
            fx1 = f(x1)
            counter += 1
        golden_sec_method_points.append([l, r])

        if L < eps:
            break

    print("Golden section method")
    print("Number of iteratioons %d" % i)
    findMin(fx1, fx2, x1, x2)
    print("Number of functions calculated: %d" % counter)
    print("***********************")


def findMin(a, b, x1, x2):
    if a < b:
        print("f(x1) =", a)
        print("x1 =", x1)
    else:
        print("f(x2) =", b)
        print("x2 =", x2)

def newtons_method(func, x0, eps):
    counter = 0

    for i in range(1, 1000):
        xn = x0 - first_deriv(x0, func) / second_deriv(x0, func)
        counter += 2
        newton_method_points.append(xn)
        if abs(xn - x0) < eps:
            break
        x0 = xn

    print("Newton's method")
    print("Number of iterations: %d" % i)
    print("xn: %f" % xn)
    print("f(xn): %f" %f(xn, func))
    print("Number of functions calculated: %d" % counter)
    print("**********************")

def f(x, func):
    return eval(func)

def first_deriv(x, func):
    h = 1e-5
    return (f(x + h, func) - f(x, func)) / h

def second_deriv(x, func):
    h = 1e-5
    return (first_deriv(x + h, func) - first_deriv(x, func)) / h

def generatePoints(point_array):
    for i in range(0, len(point_array)):
        r = random.random()
        b = random.random()
        g = random.random()
        a = 1
        color = (r, g, b, a)
        plt.scatter(point_array[i][0], 0, color=color)
        plt.scatter(point_array[i][1], 0, color=color)

        for i in range(0, len(point_array)):
            plt.annotate(i + 1, (point_array[i][0], 0))
            plt.annotate(i + 1, (point_array[i][1], 0))


def createGraph(func, char):
    x = np.linspace(-5, 6, 100)

    y = eval(func)

    print("newtons: ", newton_method_points)
    print("bisec: ", bisec_method_points)
    print("golden: ", golden_sec_method_points)

    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.set(ylim=(-5,6))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    if char == 'n':
        a = [0] * len(newton_method_points)
        plt.scatter(newton_method_points, a)
        for i in range(0, len(newton_method_points)):
            plt.annotate(i + 1, (newton_method_points[i], a[i]))
    elif char == 'b':
        generatePoints(bisec_method_points)
    elif char == 'g':
        generatePoints(golden_sec_method_points)

    plt.plot(x, y, 'r')
    plt.show()


def main():
    # function = "((x ** 2 - 4) ** 2 /9-1)"
    function = "(100 - x)**2"
    bisection_method(function, 0, 1, 0.0001)
    golden_section(function, 0, 10, 0.0001, 0.61803)
    newtons_method(function, 2, 0.0001)
    createGraph(function, 'b')


if __name__ == "__main__":
    main()
