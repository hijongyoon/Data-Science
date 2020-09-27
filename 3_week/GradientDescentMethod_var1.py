from random import random


def f(x):
    return x ** 4 - 12.0 * x ** 2 - 12.0 * x


def df(x):
    return 4 * x ** 3 - 24 * x - 12


# x(1) = x(0) - p * df(x(0)) 식을 기반으로 한 것 -> p는 적당한 상수, 기울기가 가파르면 많이 이동하고 기울기가 완만하면(0에
# 가까우면) 극소점 근처일 가능성이 있으므로 조금만 이동함.

rho = 0.005  # p를 의미
precision = 0.000000001 # 근사치
difference = 100
x = random()
# x= 4.0

while difference > precision:
    dr = df(x)
    prev_x = x # prev_x는 예전 x를 저장하는 것 즉 x(0)
    x = x - rho * dr    # 새로운 x의 값 즉 x(1)
    difference = abs(prev_x - x) # x(1) - x(0) = -p* df(x(0)) 인데 이 값이 0에 가까울 수록 극소점 즉 기울기가 0에 가까운 점에 근접하다는 의미
    print("x = {:f}, df = {:10.6f}, f(x) = {:f}".format(x, dr, f(x)))