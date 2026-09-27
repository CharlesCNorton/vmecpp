# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""How far the Landreman-Paul precise quasisymmetric equilibria are from exact
quasisymmetry on one surface, against a certified floor.

A field is quasisymmetric on a surface exactly when (B x grad s . grad B) /
(B . grad B) is constant on it (Helander 2014). Multiplied through by the Jacobian
and written against B^2, which needs no square root, the claim is t1 = lam t2 with

    t1 = L_v A_u - L_u A_v,   t2 = J C,

L_i = J B_i the Jacobian-weighted covariant components, A_i = J^3 d_i B^2 and
C = J^4 B . grad B^2. This file runs a configuration of Landreman and Paul (2021),
the reactor-scale quasi-helically symmetric one ("QH") or the quasi-axisymmetric
one ("QA"), reads R, Z, lambda and iota at the half point outside a node by VMEC's
half-grid rule, evaluates t1 and t2 at 32 x 16 angles over one field period, and
returns

    D = min over lam of max over the points of |t1 - lam t2|,

the defect of the best constant.

Stellarocq (https://github.com/CharlesCNorton/stellarocq, gen/qs_floor.py,
theories/QSFloor.v) proves a floor under D over every state whose R, Z and lambda
coefficients on that node's stencil lie within a relative 1e-8 of this run's: at two
kernels the harmonics of t1 and t2 over the points have ratios whose certified
enclosures are disjoint, which QSFloor.qs_floor turns into |t1 - lam t2| >= floor
at some point for every lam. tests/test_quasisymmetry_floor.py holds the floors and
checks that D does not fall below them.

Usage:
    python quasisymmetry_floor.py [QH|QA]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import vmecpp

DATA = Path(__file__).resolve().parent / "data"
# the input and the node whose outer half point the floor is certified on
CASES = {
    "QH": ("input.LandremanPaul2021_QH_reactorScale_lowres", 25),
    "QA": ("input.LandremanPaul2021_QA", 100),
}
NODE = CASES["QH"][1]
NU, NV = 32, 16


def run(perturb: float = 0.0, seed: int = 0, case: str = "QH"):
    """A configuration's input, its boundary optionally perturbed by a random
    relative amount per coefficient."""
    vi = vmecpp.VmecInput.from_file(DATA / CASES[case][0])
    if perturb:
        rng = np.random.default_rng(seed)
        rbc = np.asarray(vi.rbc, dtype=float)
        zbs = np.asarray(vi.zbs, dtype=float)
        rbc = rbc * (1.0 + perturb * rng.uniform(-1.0, 1.0, rbc.shape))
        zbs = zbs * (1.0 + perturb * rng.uniform(-1.0, 1.0, zbs.shape))
        vi = vi.model_copy(update={"rbc": rbc, "zbs": zbs}, deep=True)
    return vmecpp.run(vi, verbose=False).wout


def _rows(wout, name):
    a = np.asarray(getattr(wout, name), dtype=float)
    return a.T if a.shape[0] == len(np.asarray(wout.xm)) else a


def stencil(wout, j=NODE):
    """The coefficients the two terms read: R and Z on nodes j and j + 1 and
    lambda on the half point between them, as one vector."""
    R, Z, L = _rows(wout, "rmnc"), _rows(wout, "zmns"), _rows(wout, "lmns")
    return np.concatenate(
        [R[j - 1], R[j], R[j + 1], Z[j - 1], Z[j], Z[j + 1], L[j], L[j + 1]]
    )


def terms(wout, j=NODE, nu=NU, nv=NV):
    """t1 and t2 at the half point outside node j, over a grid of one period."""
    xm = np.asarray(wout.xm, dtype=float)
    xn = np.asarray(wout.xn, dtype=float)
    nfp = int(wout.nfp)
    ns = int(wout.ns)
    R, Z, L = _rows(wout, "rmnc"), _rows(wout, "zmns"), _rows(wout, "lmns")
    h = 1.0 / (ns - 1)
    sa, sb, sh = j * h, (j + 1) * h, (j + 0.5) * h
    odd = (xm % 2) == 1

    def half(ya, yb):
        ev, ed = 0.5 * (ya + yb), (yb - ya) / (sb - sa)
        qa, qb = ya / np.sqrt(sa), yb / np.sqrt(sb)
        ov = np.sqrt(sh) * 0.5 * (qa + qb)
        od = np.sqrt(sh) * (qb - qa) / (sb - sa) + ov / (2.0 * sh)
        return np.where(odd, ov, ev), np.where(odd, od, ed)

    cR, dR = half(R[j], R[j + 1])
    cZ, dZ = half(Z[j], Z[j + 1])
    lam = L[j + 1]
    iota = float(np.asarray(wout.iotas)[j + 1])
    phip = float(np.asarray(wout.phips)[1])
    u = 2 * np.pi * np.arange(nu) / nu
    v = 2 * np.pi * np.arange(nv) / (nfp * nv)
    U, V = np.meshgrid(u, v, indexing="ij")
    ang = xm[:, None, None] * U[None] - xn[:, None, None] * V[None]
    C, S = np.cos(ang), np.sin(ang)

    def cos_series(c, d):
        return {
            "0": np.tensordot(c, C, 1),
            "s": np.tensordot(d, C, 1),
            "u": np.tensordot(-xm * c, S, 1),
            "v": np.tensordot(xn * c, S, 1),
            "su": np.tensordot(-xm * d, S, 1),
            "sv": np.tensordot(xn * d, S, 1),
            "uu": np.tensordot(-xm * xm * c, C, 1),
            "uv": np.tensordot(xm * xn * c, C, 1),
            "vv": np.tensordot(-xn * xn * c, C, 1),
        }

    def sin_series(c, d):
        return {
            "0": np.tensordot(c, S, 1),
            "s": np.tensordot(d, S, 1),
            "u": np.tensordot(xm * c, C, 1),
            "v": np.tensordot(-xn * c, C, 1),
            "su": np.tensordot(xm * d, C, 1),
            "sv": np.tensordot(-xn * d, C, 1),
            "uu": np.tensordot(-xm * xm * c, S, 1),
            "uv": np.tensordot(xm * xn * c, S, 1),
            "vv": np.tensordot(-xn * xn * c, S, 1),
        }

    r = cos_series(cR, dR)
    z = sin_series(cZ, dZ)
    lm = sin_series(lam, np.zeros_like(lam))
    tau = r["u"] * z["s"] - r["s"] * z["u"]
    J = r["0"] * tau
    tau_u = r["uu"] * z["s"] + r["u"] * z["su"] - (r["su"] * z["u"] + r["s"] * z["uu"])
    tau_v = r["uv"] * z["s"] + r["u"] * z["sv"] - (r["sv"] * z["u"] + r["s"] * z["uv"])
    J_u = r["u"] * tau + r["0"] * tau_u
    J_v = r["v"] * tau + r["0"] * tau_v
    guu = r["u"] ** 2 + z["u"] ** 2
    guv = r["u"] * r["v"] + z["u"] * z["v"]
    gvv = r["v"] ** 2 + z["v"] ** 2 + r["0"] ** 2
    guu_u = 2 * (r["u"] * r["uu"] + z["u"] * z["uu"])
    guu_v = 2 * (r["u"] * r["uv"] + z["u"] * z["uv"])
    guv_u = r["uu"] * r["v"] + r["u"] * r["uv"] + z["uu"] * z["v"] + z["u"] * z["uv"]
    guv_v = r["uv"] * r["v"] + r["u"] * r["vv"] + z["uv"] * z["v"] + z["u"] * z["vv"]
    gvv_u = 2 * (r["v"] * r["uv"] + z["v"] * z["uv"] + r["0"] * r["u"])
    gvv_v = 2 * (r["v"] * r["vv"] + z["v"] * z["vv"] + r["0"] * r["v"])
    Uc = phip * (iota - lm["v"])
    Vc = phip * (1.0 + lm["u"])
    U_u, U_v = -phip * lm["uv"], -phip * lm["vv"]
    V_u, V_v = phip * lm["uu"], phip * lm["uv"]
    Lu = guu * Uc + guv * Vc
    Lv = guv * Uc + gvv * Vc
    Lu_u = guu_u * Uc + guu * U_u + guv_u * Vc + guv * V_u
    Lu_v = guu_v * Uc + guu * U_v + guv_v * Vc + guv * V_v
    Lv_u = guv_u * Uc + guv * U_u + gvv_u * Vc + gvv * V_u
    Lv_v = guv_v * Uc + guv * U_v + gvv_v * Vc + gvv * V_v
    M = Uc * Lu + Vc * Lv
    M_u = U_u * Lu + Uc * Lu_u + V_u * Lv + Vc * Lv_u
    M_v = U_v * Lu + Uc * Lu_v + V_v * Lv + Vc * Lv_v
    A_u = M_u * J - 2 * M * J_u
    A_v = M_v * J - 2 * M * J_v
    Cq = Uc * A_u + Vc * A_v
    return (Lv * A_u - Lu * A_v).ravel(), (J * Cq).ravel()


def best_defect(t1, t2):
    """min over lam of max |t1 - lam t2|, which is convex in lam."""
    lam0 = float(t1 @ t2 / (t2 @ t2))
    lo, hi = lam0 - 1.0 * abs(lam0) - 1.0, lam0 + 1.0 * abs(lam0) + 1.0

    def f(x):
        return float(np.abs(t1 - x * t2).max())

    for _ in range(200):
        a = lo + (hi - lo) / 3.0
        b = hi - (hi - lo) / 3.0
        if f(a) <= f(b):
            hi = b
        else:
            lo = a
    lam = 0.5 * (lo + hi)
    return f(lam), lam


def main():
    for case in sys.argv[1:] or list(CASES):
        node = CASES[case][1]
        w = run(case=case)
        t1, t2 = terms(w, j=node)
        D, lam = best_defect(t1, t2)
        print(
            f"{case}: node {node}, s = {(node + 0.5) / (int(w.ns) - 1):.4f}: "
            f"max |t1| {np.abs(t1).max():.3e}, max |t2| {np.abs(t2).max():.3e}, "
            f"best lam {lam:.6e}, defect D = {D:.6e}"
        )


if __name__ == "__main__":
    main()
