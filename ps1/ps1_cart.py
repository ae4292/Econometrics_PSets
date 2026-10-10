"""PS1, Problem 2(d): best depth-one CART for y = f(x) + x*eps.

x, eps ~ N(0,1) independent, f(x) = max(x, 0).  For a step function
g(x) = a if x <= x0 else b,
    MSE(g) = E[(y - E(y|x))^2] + E[(f(x) - g(x))^2] = 1 + E[(f - g)^2].
For fixed x0 the optimal a, b are the conditional means of f on each side;
x0 is found by 1-D numerical minimisation.  Deterministic, no RNG.

Run: python ps1_cart.py
"""
import numpy as np
from scipy import integrate, optimize, stats

phi, Phi = stats.norm.pdf, stats.norm.cdf
f = lambda x: np.maximum(x, 0.0)


def cart_mse(x0):
    a = integrate.quad(lambda x: f(x) * phi(x), -np.inf, x0)[0] / Phi(x0)
    b = integrate.quad(lambda x: f(x) * phi(x), x0, np.inf)[0] / (1 - Phi(x0))
    e = (integrate.quad(lambda x: (f(x) - a) ** 2 * phi(x), -np.inf, x0)[0]
         + integrate.quad(lambda x: (f(x) - b) ** 2 * phi(x), x0, np.inf)[0])
    return 1 + e, a, b


res = optimize.minimize_scalar(lambda x: cart_mse(x)[0], bounds=(-2, 3),
                               method="bounded", options={"xatol": 1e-10})
mse, a, b = cart_mse(res.x)
print(f"x0 = {res.x:.4f}   a = {a:.4f}   b = {b:.4f}")
print(f"MSE: CEF = 1.0000   CART = {mse:.4f}   BLP = {5/4 - 1/(2*np.pi):.4f}")
