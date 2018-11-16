import pandas as pd
import numpy as np


class A:
    def __init__(self, n):
        self.n = n


class B(A):
    def __init__(self, k):
        self.k = k

    def printing(self, a):
        print(self.n)

l = B(4)
l.printing(3)
