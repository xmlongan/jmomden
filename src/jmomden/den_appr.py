from collections.abc import Iterable
import numpy as np

from gramschmidt import GramSchmidt
from pearsondist.pearson8 import Pearson8


class DenAppr:
    """Density approximation for the transformed 2D variables"""

    mu_d1: list = None
    mu_d2: list = None
    mu_d1d2: list = None
    degree: int = None
    basis_d1: list = None
    basis_d2: list = None
    onb_coef: np.array = None  # orthonormal basis coefficients
    pearson_d1: Pearson8 = None
    pearson_d2: Pearson8 = None

    def __init__(self, mu_d1: list, mu_d2: list, mu_d1d2: list, degree=4):
        self.mu_d1 = mu_d1
        self.mu_d2 = mu_d2
        self.mu_d1d2 = mu_d1d2
        self.degree = degree
        gram_schmidt_d1 = GramSchmidt(degree=degree, moment=mu_d1)
        gram_schmidt_d2 = GramSchmidt(degree=degree, moment=mu_d2)
        self.basis_d1 = gram_schmidt_d1.basis
        self.basis_d2 = gram_schmidt_d2.basis
        self.pearson_d1 = Pearson8(self.mu_d1)
        self.pearson_d2 = Pearson8(self.mu_d2)
        self.comp_coef()

    def auxi_den(self, zeta1, zeta2):
        w1 = self.pearson_d1.pdf(zeta1)
        w2 = self.pearson_d2.pdf(zeta2)
        w = w1 * w2
        return w

    def comp_coef(self):
        n, m = len(self.basis_d1), len(self.basis_d2)
        onb_c = []
        for i in range(n):
            bi = self.basis_d1[i]
            c_row = []
            for j in range(m):
                bj = self.basis_d2[j]
                c_s = sum([bi[r] * bj[c] * self.mu_d1d2[r][c]
                           for r in range(n) for c in range(m)])
                c_row.append(c_s)
            onb_c.append(c_row)
        self.onb_coef = np.array(onb_c)

    def like_ratio(self, zeta1, zeta2):
        # ratio = 0
        # for i in range(len(self.basis_d1)):
        #     bi = self.basis_d1[i]  # coef: [1, zeta1, zeta1^2, zeta1^3, zeta1^4]
        #     Hi = bi.eval(zeta1)
        #     for j in range(len(self.basis_d2)):
        #         bj = self.basis_d2[j]
        #         Hj = bj.eval(zeta2)
        #         ratio += self.onb_coef[i][j] * Hi * Hj
        bc = np.array([[bi.eval(zeta1)] for bi in self.basis_d1])
        br = np.array([[bj.eval(zeta2) for bj in self.basis_d2]])
        H = bc @ br
        ratio = np.sum(self.onb_coef * H)
        return ratio

    def pseu_den(self, zeta1, zeta2):
        flag1 = isinstance(zeta1, Iterable)
        flag2 = isinstance(zeta2, Iterable)
        if not flag1 and not flag2:
            w = self.auxi_den(zeta1, zeta2)
            ratio = self.like_ratio(zeta1, zeta2)
        elif not flag1 and flag2:
            w = self.auxi_den(zeta1, zeta2)
            ratio = np.array([self.like_ratio(zeta1, zeta2[i]) for i in range(len(zeta2))])
        elif flag1 and not flag2:
            w = self.auxi_den(zeta1, zeta2)
            ratio = np.array([self.like_ratio(zeta1[i], zeta2) for i in range(len(zeta1))])
        else:
            w = self.auxi_den(zeta1, zeta2)
            ratio = np.array([self.like_ratio(zeta1[i], zeta2[i]) for i in range(len(zeta1))])
        return w * ratio

    def print_moment(self):
        mu_d1 = self.mu_d1
        mu_d2 = self.mu_d2
        print('\nMoments for the transformed variables:')
        mu_d1d2 = self.mu_d1d2
        print(f'mu_d1d2: ')
        for i in range(len(mu_d1d2)):
            n = len(mu_d1d2[i])
            txt = ",".join([f"{m:>10.7f}" for m in mu_d1d2[i]])
            print(f'mu[{i}][0:{n - 1}] = [{txt}]')

        txt = ','.join(f'{i:>10d}' for i in range(1, len(mu_d1) + 1))
        print(f'order = [{txt}]')
        txt = ",".join([f"{mu:>10.7f}" for mu in mu_d1])
        print(f'mu_d1 = [{txt}]')
        txt = ",".join([f"{mu:>10.7f}" for mu in mu_d2])
        print(f'mu_d2 = [{txt}]')

    def print_basis(self):
        basis_d1 = self.basis_d1
        basis_d2 = self.basis_d2
        print(f'\nTwo 1D orthonormal basis polynomial coefficients:')
        title1 = ', '.join(f'zeta1^{i}' for i in range(len(basis_d1[0])))
        title2 = ', '.join(f'zeta2^{i}' for i in range(len(basis_d2[0])))

        print(f'basis_d1:')
        print(f'coef:   [{title1}]')
        for i, b in enumerate(basis_d1):
            txt = ",".join([f"{b[i]:>10.7f}" for i in range(len(b))])
            print(f'basis{i}: [{txt}]')
        print(f'basis_d2:')

        print(f'coef:   [{title2}]')
        for i, b in enumerate(basis_d2):
            txt = ",".join([f"{b[i]:>10.7f}" for i in range(len(b))])
            print(f'basis{i}: [{txt}]')

    def print_onb_coef(self):
        c = self.onb_coef
        print('\nCoefficients × orthonormal basis of 2D polynomials:')
        for i in range(len(c)):
            print(', '.join(f'{c[i][j]:12.9f}' for j in range(len(c[i]))))
