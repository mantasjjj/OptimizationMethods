from sympy import Symbol


def f(x1, x2, x3):
    return -1 * x1 * x2 * x3


def gradientFunction(func, x1, x2, x3):
    return eval(func)


def getGrad(func, args):
    grad = []
    for x in args:
        grad.append(func.diff(x))
    return grad


def gradient_descent(func, args, X0, gama, epsilon):
    i = 1
    Xi = X0

    grad = getGrad(func, args)
    counter = 0

    max_iterations = 5000
    while i < max_iterations:
        gradMod = 0
        Xtemp = list(Xi)
        for j in range(0, len(Xi)):
            gradFunc = gradientFunction(str(grad[j]), Xtemp[0], Xtemp[1], Xtemp[2])
            Xi[j] = Xi[j] - gama * gradFunc
            counter += 1
            gradMod += gradFunc
        gradMod = abs(gradMod) / len(Xi)
        print("i: ", i, "Xi[0]: ", Xi[0], "Xi[1]:", Xi[1], "Xi[2]:", Xi[2])
        if gradMod < epsilon:
            print("i: ", i, "[", Xi[0], ",", Xi[1], ",", Xi[2], "]")
            print("Counter: ", counter)
            print("f(X) = ", f(Xi[0], Xi[1], Xi[2]))
            break
        i += 1

    return Xi


def optimization(args, epsilon):
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')

    func = -1 * x1 * x2 * x3

    maxIterations = 100
    # Kaip pasirenkame r?
    r = 1
    old = f(args[0], args[1], args[2])
    for i in range(1, maxIterations):
        # f(X) + 1/r * b(X)
        bfunc = func + 1 / r * penaltyFunction(args)

        args = gradientFunction(bfunc, [x1, x2, x3], args, 0.001)

        new = f(args[0], args[1], args[2])
        if abs(new - old) < epsilon:
            break
        else:
            old = new

    print("X:", args, "\nf(X):", f(args[0], args[1], args[2]), "\niterations:", i)


def penaltyFunction(args):
    g = (2 * args[0] * args[1] + 2 * args[0] * args[2] + 2 * args[2] * args[1]) - 1
    # Kaip gauti h kaip skaiciu? Nes h yra nelygybe, t.y. hj <= 0
    h = [0]

    gsum = 0
    hsum = 0
    for i in range(0, len(g)):
        gsum += g[i] ** 2

    for j in range(0, len(h)):
        hsum += min(0, h[j]) ** 2

    return gsum + hsum


def main():
    print("")


if __name__ == "__main__":
    main()
