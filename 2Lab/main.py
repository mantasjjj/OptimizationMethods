import copy

import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol


def plot3d():
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    plt.show()


def getGrad(func, args):
    grad = []
    for x in args:
        grad.append(func.diff(x))
    return grad


def gradientFunction(func, x1, x2):
    return eval(func)


def gradient_descent(func, args, X0, gama, epsilon):
    i = 0
    Xi = X0

    grad = getGrad(func, args)
    points = []

    max_iterations = 5000
    while i < max_iterations:
        gradMod = 0
        Xtemp = list(Xi)
        points.append(tuple(Xtemp + []))
        for j in range(0, len(Xi)):
            gradFunc = gradientFunction(str(grad[j]), Xtemp[0], Xtemp[1])
            Xi[j] = Xi[j] - gama * gradFunc
            gradMod += gradFunc
        gradMod = abs(gradMod) / len(Xi)

        if gradMod < epsilon:
            print("i: ", i, "X0: ", ["%0.4f" % u for u in Xi])
            break
        i += 1

    plot3d()


def newtons_method(func, x0, eps):
    for i in range(1, 1000):
        if abs(x0) < eps:
            break
        xn = x0 - first_deriv(x0, func) / second_deriv(x0, func)
        if abs(xn - x0) < eps:
            break
        x0 = xn
    return x0


def fnetwons(x, func):
    return eval(func)


def first_deriv(x, func):
    h = 1e-5
    return (fnetwons(x + h, func) - fnetwons(x, func)) / h


def second_deriv(x, func):
    h = 1e-5
    return (first_deriv(x + h, func) - first_deriv(x, func)) / h


def steepest_descent(func, args, X0, epsilon):
    i = 0
    Xi = X0
    max_iterations = 5000
    grad = getGrad(func, args)

    while i < max_iterations:
        gradMod = 0
        Xtemp = list(Xi)
        for j in range(0, len(Xi)):
            gradFunc = gradientFunction(str(grad[j]), Xtemp[0], Xtemp[1])
            gamaFunc = func
            gama = Symbol('x')
            for k in range(0, len(Xtemp)):
                gamaFunc = gamaFunc.subs(args[k], (Xtemp[k] - gama * gradFunc))

            gama_min = abs(newtons_method(str(gamaFunc), abs(sum(Xtemp) / len(Xtemp)), epsilon))

            Xi[j] = Xi[j] - gama_min * gradFunc
            gradMod += gradFunc
        gradMod = abs(gradMod) / len(Xi)

        i += 1
        if gradMod < epsilon:
            print("i: ", i, "X0: ", ["%0.6f" % u for u in Xi])
            break


def nelder_mead(f, x_start,
                step=0.1, no_improve_thr=10e-6,
                no_improv_break=10, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    '''
        @param f (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
        @param x_start (numpy array): initial position
        @param step (float): look-around radius in initial step
        @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
        @max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        @alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)
        return: tuple (best parameter array, best score)
    '''

    # init
    dim = len(x_start)
    prev_best = f(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]

    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step
        score = f(x)
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0]

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res) - 1)

        # reflection
        xr = x0 + alpha * (x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma * (x0 - res[-1][0])
            escore = f(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho * (x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma * (tup[0] - x1)
            score = f(redx)
            nres.append([redx, score])
        res = nres


def main():
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    # F = (-((1 - 2 * x1 * x2) / 2 * (x1 + x2))) ** 2 * x1 ** 2 * x2 ** 2
    # F = -((1-x1-x2)*x1*x2)
    F = -(0.125*x1*x2*(1-x1-x2))

    # gradient_descent(F, [x1, x2], [0, 0], 0.01, 0.001)
    gradient_descent(F, [x1, x2], [1, 1], 0.01, 0.001)
    gradient_descent(F, [x1, x2], [0.4, 0.9], 0.01, 0.001)

    # steepest_descent(F, [x1, x2], [0, 0], 0.001)
    steepest_descent(F, [x1, x2], [1, 1], 0.001)
    steepest_descent(F, [x1, x2], [0.4, 0.9], 0.001)

    # def f(x):
    #     return (-((1 - 2 * x[0] * x[1]) / 2 * (x[0] + x[1]))) ** 2 * x[0] ** 2 * x[1] ** 2

    def f(x):
        return ((1+x[0]+x[1])*(-x[0])*(-x[1]))

    # print(nelder_mead(f, np.array([0, 0])))
    # print(nelder_mead(f, np.array([1, 1])))
    # print(nelder_mead(f, np.array([0.4, 0.9])))


if __name__ == "__main__":
    main()
