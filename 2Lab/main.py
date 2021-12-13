import itertools
from math import sin
from operator import itemgetter
import random

import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol

fig, ax = plt.subplots()


def f(x1, x2):
    return -1 * (0.125 * x1 * x2 * (1 - x1 - x2))


def plot2d(points, iterations, method):
    delta = 0.001
    x = np.arange(0, 1.7, delta)
    y = np.arange(0, 1.7, delta)
    X, Y = np.meshgrid(x, y)
    Z = (-1) * (0.125 * X * Y * (1 - X - Y))
    CS = ax.contour(X, Y, Z, 60, linewidths=0.3)
    ax.clabel(CS, inline=True, fontsize=5)
    rangeNeeded = int(len(points) / 2)
    rangeNeededSimplex = int(len(points) / 3)
    i = 0
    if method == 'gd':
        for i in range(0, rangeNeeded):
            if i + 2 < rangeNeeded and i % 2 == 0:
                ax.plot([points[i], points[i + 1]], [points[i], points[i + 1]], marker='.')
                plt.annotate(iterations[i], (points[i], points[i] + 0.05))
                plt.annotate(iterations[i + 1], (points[i + 1], points[i + 1] + 0.05))
            elif i + 2 == rangeNeeded:
                ax.plot([points[i], points[i + 1]], [points[i], points[i + 1]], marker='x')
                plt.annotate(iterations[i], (points[i], points[i] + 0.05))
                plt.annotate(iterations[i + 1], (points[i + 1], points[i + 1] + 0.05))
    elif method == 'sd':
        while i < rangeNeeded:
            if i + 1 < rangeNeeded:
                ax.plot(points[i], points[i], marker='.')
                ax.plot(points[i + 1], points[i + 1], marker='.')
                plt.annotate(i + 1, (points[i] + 0.01, points[i] + 0.02))
                plt.annotate(i + 1, (points[i + 1], points[i + 1]))

                ax.plot(points[i+2], points[i+2], marker='.')
                ax.plot(points[i + 3], points[i + 3], marker='.')
                plt.annotate(i + 2, (points[i+2] + 0.01, points[i+2] + 0.02))
                plt.annotate(i + 2, (points[i + 3], points[i + 3]))

                i += 2
            elif i + 1 == rangeNeeded:
                ax.plot(points[i], points[i], marker='x')
                ax.plot(points[i + 1], points[i + 1], marker='x')
                plt.annotate(i + 1, (points[i] + 0.01, points[i] + 0.02))
                plt.annotate(i + 1, (points[i + 1], points[i + 1]))
                i += 2
    elif method == 'simplex' and len(points) % 3 == 0:
        for i in range(0, len(points), 3):
            simp = [points[i], points[i + 1], points[i + 2]]
            r = random.random()
            b = random.random()
            g = random.random()
            a = 1
            color = (r, g, b, a)
            for a, b in itertools.product(simp, simp):
                x = np.linspace(a[0], b[0], 100)
                y = np.linspace(a[1], b[1], 100)

                ax.plot(x, y, color=color)
                # plt.annotate(i+1, (a[0], b[0]))
                # plt.annotate(i+1, (a[1], b[1]))
                # plt.annotate(i, (x, y))
    #             plt.annotate(i + 1, (points[i][0], points[i][1]))
    #             plt.annotate(i + 1, (points[i+1][0], points[i + 1][1]))
    #             plt.annotate(i + 1, (points[i+2][0], points[i + 2][1]))
    plt.draw()
    plt.show()


def getGrad(func, args):
    grad = []
    for x in args:
        grad.append(func.diff(x))
    return grad


def gradientFunction(func, x1, x2):
    return eval(func)


def gradient_descent(func, args, X0, gama, epsilon):
    i = 1
    Xi = X0

    grad = getGrad(func, args)
    points = [Xi[0], Xi[1]]
    iterations = [0]
    counter = 0

    max_iterations = 5000
    while i < max_iterations:
        gradMod = 0
        Xtemp = list(Xi)
        for j in range(0, len(Xi)):
            gradFunc = gradientFunction(str(grad[j]), Xtemp[0], Xtemp[1])
            Xi[j] = Xi[j] - gama * gradFunc
            counter += 1
            gradMod += gradFunc
        gradMod = abs(gradMod) / len(Xi)
        iterations.append(i)
        points.append(Xi[0])
        points.append(Xi[1])
        print("i: ", i, "Xi[0]: ", Xi[0], "Xi[1]:", Xi[1])
        if gradMod < epsilon:
            print("i: ", i, "[", Xi[0], ",", Xi[1], "]")
            print("Counter: ", counter)
            print("f(X) = ", f(Xi[0], Xi[1]))
            iterations.append(i)
            points.append(Xi[0])
            points.append(Xi[1])
            break
        i += 1
    plot2d(points, iterations, 'gd')


def newtons_method(func, x0, eps):
    diff1 = func.diff('x')
    diff2 = diff1.diff('x')
    counter = 0

    for i in range(1, 1000):
        if abs(x0) < eps:
            break
        xn = x0 - (diff1.subs('x', x0) / diff2.subs('x', x0))
        counter += 3
        if abs(xn - x0) < eps:
            break
        x0 = xn
    return x0, counter


def steepest_descent(func, args, X0, epsilon):
    i = 1
    Xi = X0
    max_iterations = 5000
    grad = getGrad(func, args)
    points = [Xi[0], Xi[1]]
    counter = 0

    while i < max_iterations:
        gradMod = 0
        Xtemp = list(Xi)
        print("i: ", i, "[", Xi[0], ",", Xi[1], "]")
        for j in range(0, len(Xi)):
            gradFunc = gradientFunction(str(grad[j]), Xtemp[0], Xtemp[1])
            gamaFunc = func
            gama = Symbol('x')
            for k in range(0, len(Xtemp)):
                gamaFunc = gamaFunc.subs(args[k], (Xtemp[k] - gama * gradFunc))

            gama_min, n_count = newtons_method(gamaFunc, abs(sum(Xtemp) / len(Xtemp)), epsilon)
            counter += n_count

            Xi[j] = Xi[j] - gama_min * gradFunc
            print(i, "gamma", gama_min)
            counter += 1
            gradMod += gradFunc
        gradMod = abs(gradMod) / len(Xi)
        points.append(Xi[0])
        points.append(Xi[1])
        if gradMod < epsilon:
            print("i: ", i, "[", Xi[0], ",", Xi[1], "]")
            print("Counter: ", counter)
            print("f(X) = ", f(Xi[0], Xi[1]))
            break
        i += 1
    plot2d(points, 0, 'sd')


def getPoint(arg, value):
    return {"arg": arg, "value": value}


def simplex_method(args, X0, epsilon=0.0000001, alpha=0.5, beta=0.5, gama=3, ro=0.5):
    simplex = [getPoint(X0, f(X0[0], X0[1]))]
    max_iterations = 2000
    counter = 0
    points = []

    for i in range(0, len(args)):
        argList = list(X0)
        argList[i] += alpha
        simplex.append(getPoint(argList, f(argList[0], argList[1])))

    for i in range(0, max_iterations):
        # 1. Sort
        simplex.sort(key=itemgetter('value'))

        print("i: ", i + 1, simplex[0]['arg'])

        # if i < 6:
        points.extend([tuple(sim['arg'] + [sim['value']]) for sim in simplex])

        # 6. Check convergence
        if abs(simplex[0]['value'] - simplex[-1]['value']) < epsilon:
            break

        centroid = [0] * len(args)
        for j in range(0, len(args)):
            for k in range(0, len(simplex) - 1):
                centroid[j] += simplex[k]['arg'][j]
            centroid[j] /= (len(simplex) - 1)

        # 2. Reflect
        reflection = [0] * len(args)
        for j in range(0, len(args)):
            reflection[j] = centroid[j] + alpha * (centroid[j] - simplex[-1]['arg'][j])
        reflection_value = f(reflection[0], reflection[1])
        counter += 1

        # 3. Evaluate or Extend
        if simplex[0]['value'] <= reflection_value < simplex[-2]['value']:
            simplex[-1] = getPoint(reflection, reflection_value)
            continue
        elif reflection_value < simplex[0]['value']:
            extend = [0] * len(args)
            for j in range(0, len(args)):
                extend[j] = centroid[j] + gama * (reflection[j] - centroid[j])
            extended_value = f(extend[0], extend[1])
            counter += 1
            if extended_value < simplex[0]['value']:
                simplex[-1] = getPoint(extend, extended_value)
            else:
                simplex[-1] = getPoint(reflection, reflection_value)
            continue

        # 4. Contract
        contraction = [0] * len(args)
        for j in range(0, len(args)):
            contraction[j] = centroid[j] + ro * (simplex[-1]['arg'][j] - centroid[j])
        contraction_value = f(contraction[0], contraction[1])
        counter += 1
        if contraction_value < simplex[-1]['value']:
            simplex[-1] = getPoint(contraction, contraction_value)
            continue

        # 5. Reduce
        reduce = [0] * len(args)
        for j in range(1, len(simplex)):
            for k in range(0, len(args)):
                reduce[k] = simplex[0]['arg'][k] + beta * (simplex[j]['arg'][k] - simplex[0]['arg'][k])
            reduce_value = f(reduce[0], reduce[1])
            counter += 1
            simplex[j] = getPoint(reduce, reduce_value)

    plot2d(points, 0, 'simplex')
    print("i: ", i + 1, simplex[0]['arg'])
    print("f(X) = ", f(simplex[0]['arg'][0], simplex[0]['arg'][1]))
    print("Counter: ", counter)


def main():
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    F = -1 * (0.125 * x1 * x2 * (1 - x1 - x2))

    grad = getGrad(F, [x1, x2])

    # print("Tikslo ir gradiento funkciju reiksmes (0, 0)", f(0, 0), gradientFunction(str(grad), 0, 0))
    # print("Tikslo ir gradiento funkciju reiksmes (1, 1)", f(1, 1), gradientFunction(str(grad), 1, 1))
    # print("Tikslo ir gradiento funkciju reiksmes (0.4, 0.9)", f(0.4, 0.9), gradientFunction(str(grad), 0.4, 0.9))

    # gradient_descent(F, [x1, x2], [0, 0], 0.01, 0.001)
    # gradient_descent(F, [x1, x2], [1, 1], 3, 0.001)
    # gradient_descent(F, [x1, x2], [0.4, 0.9], 3, 0.001)


    # steepest_descent(F, [x1, x2], [0, 0], 0.001)
    # steepest_descent(F, [x1, x2], [0.1, 0.1], 0.001)
    # steepest_descent(F, [x1, x2], [0.4, 0.9], 0.001)

    # simplex_method([x1, x2], [0, 0])
    simplex_method([x1, x2], [1, 1])
    # simplex_method([x1, x2], [0.4, 0.9])


if __name__ == "__main__":
    main()
