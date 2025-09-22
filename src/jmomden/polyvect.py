import math


class PolyVect:
    coef: list = None

    def __init__(self, coefficients):
        # [1, x, ..., x^n]
        # [c0, c1, ..., cn]
        self.coef = coefficients.copy()

    def __len__(self):
        return len(self.coef)

    def __getitem__(self, index):
        return self.coef[index]

    def __setitem__(self, index, value):
        self.coef[index] = value

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            poly = PolyVect(self.coef)
            for i in range(len(poly)):
                poly[i] = poly[i] / other
            return poly
        else:
            raise TypeError("Cannot divide polynomial with other type")

    def __mul__(self, other):
        if isinstance(other, PolyVect):
            n, m = len(self), len(other)
            coef = [[self[i] * other[j] for j in range(m)] for i in range(n)]
            return coef
        elif isinstance(other, (int, float)):
            poly = PolyVect(self.coef)
            for i in range(len(poly)):
                poly[i] = poly[i] * other
            return poly
        else:
            return NotImplemented

    def eval(self, x):
        return sum(self[i] * (x ** i) for i in range(len(self)))

    def __str__(self):
        return f"poly with coef = {self.coef}"

    def __sub__(self, other):
        if not isinstance(other, PolyVect) or len(other) != len(self):
            raise TypeError("Cannot subtract other type")
        new_poly = PolyVect(self.coef)
        for i in range(len(self)):
            new_poly[i] = self[i] - other[i]
        return new_poly

    def __rmul__(self, other):
        return self.__mul__(other)

    def scalar_product(self, other, moment: list):
        r"""Scalar product of two polynomials

        :param PolyVect other: another Polynomial [x^0, x^1, ..., x^n]
        :param list moment: moments, :math:`(\mu_1,\cdots,\mu_{2n})`
        :return: scalar value
        :rtype: float
        """
        if not isinstance(other, PolyVect):
            return NotImplemented
        if len(other) != len(self):
            raise ValueError("orders of the two polynomials must be the same")
        if len(moment) != 2 * (len(self) - 1):
            raise ValueError("number of moments must be twofold of the order")
        f = 0
        for i in range(len(self)):
            for j in range(len(other)):
                c = self[i] * other[j]
                m = 1.0 if i + j == 0 else moment[i + j - 1]
                f += c * m
                # print(f'c = {c}, m = {m}, f = {f}')
        return f

    def norm(self, moment):
        norm_squared = self.scalar_product(self, moment)
        if norm_squared <= 0:
            print(f'PolyVect: {self.coef}')
            print(f'moment = {moment}')
            print(f'norm_squared = {norm_squared}')
        # elif norm_squared <= 1e-10:
        #     print(f'PolyVect: {self.coef}')
        #     print(f'moment = {moment}')
        #     print(f'norm_squared = {norm_squared}')
        return math.sqrt(norm_squared)
