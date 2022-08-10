from e3nn.o3 import ReducedTensorProducts as rtp
from time import time


def timing(func, n, *args, **kwargs):
    tot = 0
    for _ in range(n):
        t = time()
        func(*args, **kwargs)
        tot += time() - t
    return tot / n


def time_rtp(formula, n, **irreps):
    return timing(rtp, n, formula, **irreps)


if __name__ == "__main__":
    print(time_rtp("ijk=jik=jki", 10, i="0e+1o"))