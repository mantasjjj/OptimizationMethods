from operator import itemgetter


def getPoint(arg, value):
    return {"arg": arg, "value": value}


def simplex_method(func, args, epsilon=0.0000001, alpha=0.5, beta=0.5, gama=3, ro=0.5):
    simplex = [getPoint(args, func(args))]
    max_iterations = 2000
    counter = 0
    points = []

    for i in range(0, len(args)):
        argList = list(args)
        argList[i] += alpha
        simplex.append(getPoint(argList, func(argList)))

    for i in range(0, max_iterations):
        # 1. Sort
        simplex.sort(key=itemgetter('value'))

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
        reflection_value = func(reflection)
        counter += 1

        # 3. Evaluate or Extend
        if simplex[0]['value'] <= reflection_value < simplex[-2]['value']:
            simplex[-1] = getPoint(reflection, reflection_value)
            continue
        elif reflection_value < simplex[0]['value']:
            extend = [0] * len(args)
            for j in range(0, len(args)):
                extend[j] = centroid[j] + gama * (reflection[j] - centroid[j])
            extended_value = func(extend)
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
        contraction_value = func(contraction)
        counter += 1
        if contraction_value < simplex[-1]['value']:
            simplex[-1] = getPoint(contraction, contraction_value)
            continue

        # 5. Reduce
        reduce = [0] * len(args)
        for j in range(1, len(simplex)):
            for k in range(0, len(args)):
                reduce[k] = simplex[0]['arg'][k] + beta * (simplex[j]['arg'][k] - simplex[0]['arg'][k])
            reduce_value = func(reduce[0], reduce[1])
            counter += 1
            simplex[j] = getPoint(reduce, reduce_value)


    return simplex[0]['arg']


def optimization(func, constraints, args, epsilon):
    equals = []
    inequals = []

    for con in constraints:
        if con.get("type") == "eq":
            equals.append(con.get("func"))
        elif con.get("type") == "ineq":
            inequals.append(con.get("func"))

    penaltyFunction = lambda x: sum((abs(eq(x)) ** 2) for eq in equals) + sum(
        (abs(min(0, iq(x))) ** 2) for iq in inequals)

    maxIterations = 100
    # Kaip pasirenkame r?
    r = 1
    old = func(args)
    for i in range(1, maxIterations):
        # f(X) + 1/r * b(X)
        # f(X) + 1/r * sum(g1(X)^2) + sum((max(0, h(X))^2)
        bfunc = lambda x: func(x) + 1 / r * penaltyFunction(x)

        args = simplex_method(bfunc, args)

        new = func(args)
        if abs(new - old) < epsilon:
            break
        else:
            old = new

    print("X:", args, "\nf(X):", func(args), "\niterations:", i)


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

    optimization(func, constraints, [0, 0, 0], 0.0001)
    optimization(func, constraints, [1, 1, 1], 0.0001)
    optimization(func, constraints, [0.9, 0.4, 0.9], 0.0001)

if __name__ == "__main__":
    main()
