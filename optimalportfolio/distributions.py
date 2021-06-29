import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from math import *


class BaseDist:
    """
    Provides the base for distributions class.
    """
    def __init__(self, n, mean, cov):
        self.n = n
        self.mean = mean
        self.cov = cov


class MultiNormal(BaseDist):
    def __init__(self, n, mean, cov):
        super().__init__(n, mean, cov)

    def pdf(self, x):
        """
        Calculates probability of x in multivariate_normal pdf
        :param x: value at which pdf is calculated
        :type x: nd.array
        :return: probability of x
        """
        dist = multivariate_normal(self.mean, self.cov)
        return dist.pdf(x)


class StudentT(BaseDist):
    def __init__(self, n, mean, cov, df):
        super().__init__(n, mean, cov)
        self.df = df

    def pdf(self, x, df):
        """
        Calculates the pdf for given mean, covariance and x value
        :param x: value at which pdf is calculated
        :type x: nd.array
        :param df: degrees of freedom
        :type df: int
        :return: probability of x
        """
        num = gamma(1. * (self.n + df) / 2)
        denom = (gamma(1. * df / 2) * pow(df * pi, 1. * self.n / 2) * pow(np.linalg.det(self.cov), 1. / 2) * pow(
            1 + (1. / df) * np.dot(np.dot((x - self.mean), np.linalg.inv(self.cov)), (x - self.mean)),
            1. * (self.n + df) / 2))
        prob = 1. * num / denom
        return prob


if __name__ == "__main__":
    data = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
    frame = pd.DataFrame(data)
