from e3nn.o3 import ReducedTensorProducts as rtp1
from e3nn.o3._reduce import ReducedTensorProducts as rtp2
from time import time


def timing(func, *args, n=10):
    tot = 0
    for _ in range(n):
        t = time()
        func(*args)
        tot += time() - t
    return tot / n


def test_rtp_original():
    rtp = rtp1("ij=ji", i="1o")