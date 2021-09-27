import random

import numpy as np
from matplotlib import pyplot as plt

bisec_method_points = []
newton_method_points = []
golden_sec_method_points = []


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
            l = xm
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
        bisec_method_points.append([l, r])

    print("Bisection method")
    print("Number of iterations: %d" % i)
    find_min(f(x1), f(x2), f(xm), x1, x2, xm)
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
        golden_sec_method_points.append([l, r])

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


def newtons_method(func, x0, eps):
    def f(x):
        return eval(func)

    def first_diff(x):
        h = 1e-5
        return (f(x + h) - f(x)) / h

    def second_diff(x):
        h = 1e-5
        return (first_diff(x + h) - first_diff(x)) / h

    for i in range(1, 1000):
        xn = x0 - first_diff(x0) / second_diff(x0)
        newton_method_points.append(xn)
        if abs(xn - x0) < eps:
            break
        x0 = xn

    print("Newton's method")
    print("Number of iterations: %d" % i)
    print("xn: %f" % xn)
    print("**********************")


def generatePoints(point_array):
    for i in range(0, len(point_array)):
        r = random.random()
        b = random.random()
        g = random.random()
        a = 1
        color = (r, g, b, a)
        if (i == 0 or i == 1 or i == 3 or i == 15):
            plt.scatter(point_array[i][0], 0, color=color)
            plt.scatter(point_array[i][1], 0, color=color)

        for i in range(0, len(point_array)):
            if (i == 0 or i == 1 or i == 3):
                plt.annotate(i + 1, (point_array[i][0], 0))
                plt.annotate(i + 1, (point_array[i][1], 0))
            if (i == 15):
                plt.annotate(i + 2, (point_array[i][0], 0))
                plt.annotate(i + 2, (point_array[i][1], 0))


def createGraph(func, char):
    x = np.linspace(0, 5, 100)

    y = eval(func)

    print("newtons: ", newton_method_points)
    print("bisec: ", bisec_method_points)
    print("golden: ", golden_sec_method_points)

    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.set_ylim([-2, 3])
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
    function = "((x ** 2 - 4) ** 2 /9-1)"
    bisection_method(function, 0, 10, 0.0001)
    golden_section(function, 0, 10, 0.0001, 0.61803)
    newtons_method(function, 5, 0.0001)
    createGraph(function, 'b')


if __name__ == "__main__":
    main()
