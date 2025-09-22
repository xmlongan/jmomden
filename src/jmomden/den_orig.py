import math
import warnings
from collections.abc import Iterable
import numpy as np

from den_appr import DenAppr


class DenOrig:
    """Density approximation for the original 2D variables"""

    jmom: list = None
    den_appr: DenAppr = None

    def __init__(self, jmom: list, degree: int = 4):
        self.jmom = jmom
        jmom_tr = DenOrig.joint_mom_tr
        # get joint moments of the transformed 2D variables
        mu_d1 = [jmom_tr(i, 0, jmom) for i in range(1, 2 * degree + 1)]
        mu_d2 = [jmom_tr(0, i, jmom) for i in range(1, 2 * degree + 1)]
        mu_d1d2 = [[jmom_tr(i, j, jmom) for j in range(degree + 1)] for i in range(degree + 1)]
        # mu_d1d2 = joint_mom_tr_to(degree, jmom)
        self.den_appr = DenAppr(mu_d1, mu_d2, mu_d1d2, degree)

    def transform(self, v, y):
        cov = self.jmom[1][1] - self.jmom[1][0] * self.jmom[0][1]
        var = self.jmom[2][0] - (self.jmom[1][0]) ** 2
        c = - cov / var
        zeta1 = v
        zeta2 = c * v + y
        return zeta1, zeta2

    def join_den(self, v, y):
        zeta1, zeta2 = self.transform(v, y)
        det = 1
        return self.den_appr.pseu_den(zeta1, zeta2) * abs(det)

    def cond_den(self, v, y, to_positive=True, warn=False):
        # return self.cond_den_1(y, v, to_positive, warn)
        return self.cond_den_2(y, v, to_positive, warn)

    def cond_den_1(self, y, v, to_positive=True, warn=False):
        v_den = self.den_appr.pearson_d1.pdf(v)
        den = self.join_den(v_den, y) / v_den
        return DenOrig.make_positive(den, warn) if to_positive else den

    def cond_den_2(self, y, v, to_positive=True, warn=False):
        like_ratio = self.den_appr.like_ratio

        zeta1, zeta2 = self.transform(v, y)
        flag1 = isinstance(zeta1, Iterable)
        flag2 = isinstance(zeta2, Iterable)
        w2 = self.den_appr.pearson_d2.pdf(zeta2)
        if not flag1 and not flag2:
            ratio = like_ratio(zeta1, zeta2)
        elif not flag1 and flag2:
            ratio = np.array([like_ratio(zeta1, zeta2[i]) for i in range(len(zeta2))])
        elif flag1 and not flag2:
            ratio = np.array([like_ratio(zeta1[i], zeta2) for i in range(len(zeta1))])
        else:
            ratio = np.array([like_ratio(zeta1[i], zeta2[i]) for i in range(len(zeta1))])
        den = w2 * ratio
        return DenOrig.make_positive(den, warn) if to_positive else den

    def print_moment(self):
        mu = self.jmom
        n = self.den_appr.degree
        print(f'Original joint moments of (v_t, y_t):')
        for i in range(2 * n + 1):
            txt = ",".join([f"{m:>12.9f}" for m in mu[i][0:(2 * n - i + 1)]])
            print(f'mu[{i}][0:{2 * n - i}] = [{txt}]')

    @classmethod
    def make_positive(cls, dx, warn=False):
        """Make the density positive"""
        if not isinstance(dx, Iterable):
            return dx if dx > 0 else 1e-10
        I = dx <= 0
        if sum(I) > 0:
            dx_min = min(dx[~I])
            eps = min(1e-7, dx_min)
            if warn:
                msg = f'{sum(I)} non-positive densities in cond_den: '
                msg += f'\n(min,max) = ({min(dx[I]):.7f},{max(dx[I]):.7f}) '
                msg += f'adjust them to a small positive value ({eps:.7f}).'
                warnings.warn(msg)
            dx[I] = eps
        return dx

    @classmethod
    def joint_mom_tr(cls, n, m, jmoms):
        """Joint moments of the transformed 2D variables

        :param int n: order of zeta1
        :param int m: order of zeta2
        :param jmoms: joint moments of the original 2D variables
        :return: joint moment value of the transformed 2D variables
        :rtype: float
        """
        m1 = jmoms[1][0]
        m2 = jmoms[0][1]
        cov = jmoms[1][1] - m1 * m2
        var = jmoms[2][0] - m1 ** 2
        c = - cov / var
        f = 0
        for i in range(m + 1):
            j = m - i
            bino = math.comb(m, i)
            f += bino * c ** i * jmoms[n + i][j]
        return f
