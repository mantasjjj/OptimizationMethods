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


# pritaikyt, kad su lambda veiktu
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


def optimization(func, constraints, args, epsilon):
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')

    equals = []
    inequals = []

    for con in constraints:
        if con.get("type") == "eq":
            equals.append(con.get("func"))
        elif con.get("type") == "ineq":
            inequals.append(con.get("func"))

    penaltyFunction = lambda x: sum((abs(eq(x)) ** 2) for eq in equals) + sum((abs(min(0, iq(x))) ** 2) for iq in inequals)

    maxIterations = 100
    # Kaip pasirenkame r?
    r = 1
    old = f(args[0], args[1], args[2])
    for i in range(1, maxIterations):
        # f(X) + 1/r * b(X)
        # f(X) + 1/r * sum(g1(X)^2) + sum((max(0, h(X))^2)
        bfunc = lambda x: func(x) + 1 / r * penaltyFunction(x)

        args = gradient_descent(bfunc, [x1, x2, x3], args, 3, 0.001)

        new = f(args[0], args[1], args[2])
        if abs(new - old) < epsilon:
            break
        else:
            old = new

    print("X:", args, "\nf(X):", f(args[0], args[1], args[2]), "\niterations:", i)


def main():
    func = lambda x: -1 * x[0] * x[1] * x[2]

    g1 = lambda x: (2 * x[0] * x[1] + 2 * x[0] * x[2] + 2 * x[2] * x[1]) - 1

    h1 = lambda x: x[0]
    h2 = lambda x: x[1]
    h3 = lambda x: x[2]

    constraints = (
        {"type": "eq", "func": g1},
        {"type": "ineq", "func": h1},
        {"type": "ineq", "func": h2},
        {"type": "ineq", "func": h3})

    optimization(func, constraints, [1, 1, 1], 0.001)

    print("")


if __name__ == "__main__":
    main()
