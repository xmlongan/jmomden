# jmomden

## Description

The package `jmomden` is a `Python` package designed for density approximation of 2D distributions
whose joint moments are given. 

## Simple Usage

installment (in the directory that contains `pyproject.py`):

```bash
pip install .
```

An example:

```python
from jmomden import DenOrig

# take as an example: Heston model 2D joint variables (v_t, y_t) 
# install ajdmom via: pip install git+https://github.com/xmlongan/ajdmom
from ajdmom.mdl_1fsv.cond_joint_mom import joint_mom, poly2num

# provide joint moments E[v_t^i y_t^j] with order i + j <= 8
par = {'h': 1, 'v0': 0.010201, 'k': 6.21, 'theta': 0.019,
       'sigma': 0.61, 'rho': -0.7, 'mu': 0.0319}
n = 8 # i + j = 8
joint_moments = [[poly2num(joint_mom(i, j), par) for j in range(n - i + 1)] 
                 for i in range(n + 1)]

# density approximation
den_orig = DenOrig(joint_moments, degree = 4)
den_orig.print_moment()

den_appr = den_orig.den_appr
den_appr.print_moment()
den_appr.print_basis()
den_appr.print_onb_coef()
```

## Documentation

The documentation probably would be hosted on <http://www.yyschools.com/jmomden/>

## Ongoing Development

This code is being developed on an on-going basis at the author's [Github site](https://github.com/xmlongan/jmomden).

## Support

For support in using this software, submit an [issue](https://github.com/xmlongan/jmomden/issues/new).
