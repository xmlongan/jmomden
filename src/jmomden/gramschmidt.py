from jmomden.polyvect import PolyVect


class GramSchmidt:
    """Gram Schmidt Normalization"""

    degree: int = None
    moment: list = None
    basis: list = None

    def __init__(self, degree: int, moment: list):
        r"""Initialize a GramSchmidt object.

        :param int degree: degree of the Orthonormal Basis.
        :param list moment: moments of the distribution, :math:`\mu_0` should
         not be included.
        """
        if len(moment) < 2 * degree:
            purpose = 'for computing the norm'
            msg = f"The number of moments must be at least twofold of the degree, {purpose}"
            raise ValueError(msg)
        self.degree = degree
        # pass on only the necessary many of moments, mu_1 to mu_n, n = 2*degree
        self.moment = moment[:(2 * degree)].copy()
        self.basis = []
        #             x^0, x^1, ..., x^degree
        b0 = PolyVect([1] + [0] * self.degree)
        self.basis.append(b0)
        for i in range(1, self.degree + 1):
            bi = self.next_poly(i)
            self.basis.append(bi)

    def next_poly(self, n: int) -> PolyVect:
        if len(self.basis) != n:
            raise Exception('lower-order polynomials are not ready')
        xn = PolyVect([0] * (self.degree + 1))
        xn[n] = 1  # x^n
        b = PolyVect(xn.coef)
        for i in range(n):
            c_xn_bi = xn.scalar_product(self.basis[i], self.moment)
            b = b - c_xn_bi * self.basis[i]
        return b / b.norm(self.moment)
