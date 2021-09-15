def golden_section(func, l, r, eps, t):
    def f(x):
        return eval(func)

    L = r - l
    x1 = r - t * L
    x2 = l + t * L
    for i in range(0, 100):
        if f(x2) < f(x1):
            l = x1
            L = r - l
            x1 = x2
            x2 = l + t * L
        else:
            r = x2
            L = r - l
            x2 = x1
            x1 = r - t*L

        if L < eps:
            break

    print("Number of iteratioons %d" % i)
    findMin(f(x1), f(x2), x1, x2)

def findMin(a, b, x1, x2):
    if a < b:
        print("f(x1) =", a)
        print("x1 =", x1)
    else:
        print("f(x2) =", b)
        print("x2 =", x2)

golden_section("((x ** 2 - 4) ** 2 /9)-1", 0, 10, 0.0001, 0.61803)