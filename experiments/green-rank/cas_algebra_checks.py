"""
cas_algebra_checks.py -- symbolic (non-LLM) verification of the algebra/exponent bookkeeping
in the Angular-Preservation follow-up paper. A "C3 falsification net" (PROTOCOL sense) for the
error class that recurs in LaTeX proofs: constants and t-exponents. Uses SymPy only; asserts,
so a wrong claim raises. Run: `python cas_algebra_checks.py`.

Checks:
  1. Master/distributional inequality: min of 6*phi*tau + 24*V/tau^2 is 18*(phi^2 V)^{1/3}.
  2. Heat-kernel L2 scalings that decide the d=4 dispute:
       full difference   ||p_t^g - p_t^{g0}||_2  ~ t^{-d/4}   (integrable iff d<4)
       shell-centered    N(t)                    ~ t^{1-d/4}  (integrable iff d<8)
  3. u_0 = (det g)^{-1/4} = 1 + (1/12) Ric r^2 + ...
  4. Parametrix L^p window (d/2, d/(d-2)) nonempty iff d in {2,3}.
  5. Perturbative half-time-split Duhamel envelope integrates to t^{-d/4}; shell-change (II)
     integrand gives t^{-d/2} (=> t^{-d/4}); both integrable on (0,1] iff d<=3.
"""
import sympy as sp

t, r, d, phi, V, tau, Ric, s, C = sp.symbols('t r d phi V tau Ric s C', positive=True)


def texp(prefactor_t_power, rad_extra):
    """Exponent of t in  t^{prefactor} * int_0^inf r^{d-1+rad_extra} e^{-r^2/(2t)} dr.
    Extract via  t * d/dt log(expr)  (= the power of t, robust to constant factors)."""
    radial = sp.integrate(r ** (d - 1 + rad_extra) * sp.exp(-r ** 2 / (2 * t)), (r, 0, sp.oo))
    expr = t ** prefactor_t_power * radial
    return sp.simplify(t * sp.diff(sp.log(expr), t))


def check(name, got, want):
    ok = sp.simplify(got - want) == 0
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: got {got}, want {want}")
    assert ok, f"{name}: {got} != {want}"


def main():
    print("=== 1. master-inequality constant ===")
    g = 6 * phi * tau + 24 * V / tau ** 2
    taustar = [c for c in sp.solve(sp.diff(g, tau), tau) if c.is_real][0]
    const = sp.simplify(g.subs(tau, taustar) / (phi ** 2 * V) ** sp.Rational(1, 3))
    check("min value constant", const, sp.Integer(18))

    print("=== 2. heat-kernel L2 scalings (the d=4 dispute) ===")
    # full difference: integrand^2 has prefactor (4pi t)^{-d} * (1/t^2), radial weight r^4
    full2 = texp(-d - 2, 4)              # t^{-d} * t^{-2} * (radial ~ t^{(d+4)/2})
    check("||p^g-p^{g0}||^2 exponent", sp.simplify(full2), -d / 2)      # => ||.|| ~ t^{-d/4}
    # shell-centered: prefactor (4pi t)^{-d}, radial weight r^4 (no extra 1/t^2)
    shell2 = texp(-d, 4)
    check("N(t)^2 exponent", sp.simplify(shell2), (4 - d) / 2)          # => N ~ t^{1-d/4}
    print("     => full diff ~ t^{-d/4} (int. iff d<4);  shell-centered ~ t^{1-d/4} (int. iff d<8)")

    print("=== 3. u_0 = (det g)^{-1/4} expansion ===")
    u0 = ((1 - sp.Rational(1, 6) * Ric * r ** 2) ** 2) ** sp.Rational(-1, 4)
    coeff = sp.series(u0, r, 0, 3).removeO().coeff(r, 2)
    check("Ric r^2 coefficient", coeff, Ric / 12)

    print("=== 4. parametrix L^p window nonempty iff d in {2,3} ===")
    for dd, want in [(2, True), (3, True), (4, False), (5, False)]:
        lo = sp.Rational(dd, 2)
        hi = sp.oo if dd == 2 else sp.Rational(dd, dd - 2)
        got = bool(lo < hi)
        print(f"  [{'OK ' if got == want else 'FAIL'}] d={dd}: ({lo},{hi}) nonempty={got}")
        assert got == want

    print("=== 5. perturbative half-time-split + shell-change exponents ===")
    # (0,t/2): (t-s)^{-1/2-d/4} ~ t^{-1/2-d/4}; int_0^{t/2} s^{-1/2} ds ~ t^{1/2}  => t^{-d/4}
    half = sp.Rational(-1, 2) - d / 4 + sp.Rational(1, 2)
    check("Duhamel head exponent", sp.simplify(half), -d / 4)
    # shell change (II): prefactor t^{-d-1}, radial r^{d+1}  => t^{-d/2}
    II2 = texp(-d - 1, 2)
    check("(II) integrand^2 exponent", sp.simplify(II2), -d / 2)
    print("     => N^g(t) ~ eta t^{-d/4}; int_0^1 finite iff d<=3.")

    print("\nALL SYMBOLIC CHECKS PASSED.")


if __name__ == "__main__":
    main()
