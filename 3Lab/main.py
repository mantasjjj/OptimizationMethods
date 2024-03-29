from operator import itemgetter


def getPoint(arg, value):
    return {"arg": arg, "value": value}


def simplex_method(func, args, epsilon=0.0000001, alpha=0.5, beta=0.5, gama=3, ro=0.5):
    simplex = [getPoint(args, func(args))]
    max_iterations = 2000
    counter = 0

    for i in range(0, len(args)):
        argList = list(args)
        argList[i] += alpha
        simplex.append(getPoint(argList, func(argList)))

    for i in range(0, max_iterations):
        # 1. Sort
        simplex.sort(key=itemgetter('value'))

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
            reduce_value = func(reduce)
            counter += 1
            simplex[j] = getPoint(reduce, reduce_value)

    return simplex[0]['arg'], counter


def optimization(func, constraints, args, epsilon):
    equals = []
    inequals = []
    counter = 0
    c = 0

    for con in constraints:
        if con.get("type") == "eq":
            equals.append(con.get("func"))
        elif con.get("type") == "ineq":
            inequals.append(con.get("func"))

    penaltyFunction = lambda x: sum((abs(eq(x)) ** 2) for eq in equals) + sum(
        (abs(max(0, iq(x))) ** 2) for iq in inequals)

    maxIterations = 100
    r = lambda x: x * 0.5
    old = func(args)
    penaltyQuantifier = 1
    for i in range(1, maxIterations):
        print("r: ", r(penaltyQuantifier))
        bfunc = lambda x: func(x) + 1 / r(penaltyQuantifier) * penaltyFunction(x)
        print("Baudos funkcija:", bfunc(args))
        print("F(X): ", func(args))
        penaltyQuantifier = r(penaltyQuantifier)

        args, c = simplex_method(bfunc, args)

        counter += c

        new = func(args)
        if abs(new - old) < epsilon:
            break
        else:
            old = new

    print("X:", args, "\nf(X):", func(args), "\niterations:", i)
    print("counter: ", c)


def main():
    # Budas 1: kai funkcija yra Spav, o g yra plotas - 1
    func = lambda x: -1 * (2 * x[0] * x[1] + 2 * x[0] * x[2] + 2 * x[2] * x[1])

    g1 = lambda x: x[0] + x[1] + x[2] - 1

    h1 = lambda x: x[0] - 1
    h2 = lambda x: x[1] - 1
    h3 = lambda x: x[2] - 1

    constraints = (
        {"type": "eq", "func": g1},
        {"type": "ineq", "func": h1},
        {"type": "ineq", "func": h2},
        {"type": "ineq", "func": h3})

    # print("Funkcija taske 0,0,0:", func([0, 0, 0]))
    print("Funkcija taske 1,1,1:", func([1, 1, 1]))
    # print("Funkcija taske 0.9,0.4,0.9:", func([0.9, 0.4, 0.9]))

    print("g(X) taske 0,0,0:", g1([0, 0, 0]))
    print("g(X) taske 1, 1, 1:", g1([1, 1, 1]))
    print("g(X) taske 0.9,0.4,0.9:", g1([0.9, 0.4, 0.9]))

    print("h1(X) taske 0,0,0:", h1([0, 0, 0]))
    print("h2(X) taske 0,0,0:", h2([0, 0, 0]))
    print("h3(X) taske 0,0,0:", h3([0, 0, 0]))

    print("h1(X) taske 1,1,1:", h1([1, 1, 1]))
    print("h2(X) taske 1,1,1:", h2([1, 1, 1]))
    print("h3(X) taske 1,1,1:", h3([1, 1, 1]))

    print("h1(X) taske 0.9,0.4,0.9:", h1([0.9, 0.4, 0.9]))
    print("h2(X) taske 0.9,0.4,0.9:", h2([0.9, 0.4, 0.9]))
    print("h3(X) taske 0.9,0.4,0.9:", h3([0.9, 0.4, 0.9]))

    # optimization(func, constraints, [0, 0, 0], 0.0001)
    # optimization(func, constraints, [1, 1, 1], 0.0001)
    optimization(func, constraints, [0.9, 0.4, 0.9], 0.0001)


if __name__ == "__main__":
    main()
