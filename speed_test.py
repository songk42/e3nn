from e3nn.o3 import ReducedTensorProducts as rtp
from time import time


def timing(func, *args, n=10):
    tot = 0
    for _ in range(n):
        t = time()
        func(*args)
        tot += time() - t
    return tot / n


print(timing(lambda: rtp("ij=ji", i="1o")))