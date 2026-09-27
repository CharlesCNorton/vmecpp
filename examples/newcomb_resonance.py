# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Newcomb's condition on the iota = 10/11 surface of W7-X at finite beta.

On a surface where iota = n/m the field lines close after m toroidal turns, and
in straight-field-line angles (theta*, v) the integral of dl/B along the line
theta* = alpha + iota v is

    (1 / phip) int_0^{2 pi m} sqrt(g*) dv = (2 pi m / phip) sum_p G_{pm, pn} e^{i p m alpha},

with G_{jk} the Fourier coefficients of the Jacobian sqrt(g*) of those angles. A
smooth equilibrium with nested surfaces and p' != 0 there has that integral
independent of alpha (Newcomb 1959), so the resonant coefficient G_{m,n} vanishes.

This file runs the W7-X case of the test data at finite pressure, finds the node
whose half points bracket the crossing of iota through 10/11, carries R and Z of
the three nodes around it into straight-field-line angles theta* = u + lambda,
truncated to m <= M and |n| <= N nfp, and integrates sqrt(g*) cos(11 theta* - 10 v)
over the angles at the crossing, the rows interpolated radially by VMEC's
parity-aware half-grid rule and a cubic Hermite through the two half points.

Stellarocq (https://github.com/CharlesCNorton/stellarocq, gen/newcomb.py,
theories/Newcomb.v) established that integral over every state within a relative
1e-8 of the ns = 49 reconstruction at the nominal pressure, together with the
crossing of iota and a floor on |mu0 p'| there; tests/test_newcomb_resonance.py
holds those enclosures.

Usage:
    python newcomb_resonance.py [--ns 49] [--pres-scale 1.0] [--M 24] [--N 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import vmecpp

MU_0 = 4.0e-7 * np.pi
M_RES, N_RES = 11, 10
DATA = Path(__file__).resolve().parent / "data"


def run_w7x(ns: int, pres_scale: float = 1.0, ftol: float = 1e-14):
    """The W7-X test case at ns radial points, its pressure scaled."""
    base = vmecpp.VmecInput.from_file(DATA / "w7x.json")
    steps = [x for x in (25, 49, 99, 199) if x < ns] + [ns]
    vi = base.model_copy(
        update={
            "ns_array": np.array(steps),
            "ftol_array": np.full(len(steps), ftol),
            "niter_array": np.full(len(steps), 40000),
            "pres_scale": pres_scale,
        },
        deep=True,
    )
    return vmecpp.run(vi, verbose=False).wout


class Surfaces:
    """The coefficients of a wout as radial rows, with the grid they sit on."""

    def __init__(self, wout):
        self.xm = np.asarray(wout.xm, dtype=float)
        self.xn = np.asarray(wout.xn, dtype=float)
        self.ns = int(wout.ns)
        mnmax = len(self.xm)

        def rows(a):
            a = np.asarray(a, dtype=float)
            return a.T if a.shape[0] == mnmax and a.shape[1] == self.ns else a

        self.rmnc = rows(wout.rmnc)
        self.zmns = rows(wout.zmns)
        self.lmns = rows(wout.lmns)
        self.iotas = np.asarray(wout.iotas, dtype=float)
        self.nfp = int(wout.nfp)
        h = 1.0 / (self.ns - 1)
        self.s_full = np.arange(self.ns) * h
        self.s_half = (np.arange(self.ns) - 0.5) * h  # row j of lmns and iotas


def rational_node(w: Surfaces, m: int = M_RES, n: int = N_RES):
    """The node whose two half points bracket the crossing of iota through n/m,
    and the crossing, iota linear between half points."""
    target = n / m
    for j in range(1, w.ns - 2):
        a, b = w.iotas[j], w.iotas[j + 1]
        if (a - target) * (b - target) <= 0 and a != b:
            t = (target - a) / (b - a)
            return j, w.s_half[j] + t * (w.s_half[j + 1] - w.s_half[j])
    msg = f"iota does not cross {n}/{m} on the half grid"
    raise ValueError(msg)


def pest_rows(w: Surfaces, j: int, M: int, N: int):
    """R (cos) and Z (sin) of node j on a grid of straight-field-line angles,
    transformed back to series over m <= M, |n| <= N nfp."""
    nu, nv = 4 * (M + 1), 4 * (2 * N + 1)
    th = 2 * np.pi * np.arange(nu) / nu
    vv = 2 * np.pi * np.arange(nv) / (w.nfp * nv)
    lam = 0.5 * (w.lmns[j] + w.lmns[j + 1])
    R = np.zeros((nv, nu))
    Z = np.zeros((nv, nu))
    for b, v in enumerate(vv):
        u = th.copy()
        for _ in range(60):  # u + lambda(u, v) = theta*
            ang = np.multiply.outer(w.xm, u) - w.xn[:, None] * v
            f = u + lam @ np.sin(ang) - th
            du = f / (1.0 + (lam * w.xm) @ np.cos(ang))
            u -= du
            if np.abs(du).max() < 1e-15:
                break
        ang = np.multiply.outer(w.xm, u) - w.xn[:, None] * v
        R[b] = w.rmnc[j] @ np.cos(ang)
        Z[b] = w.zmns[j] @ np.sin(ang)
    modes = [
        (m, n * w.nfp)
        for m in range(M + 1)
        for n in range(-N, N + 1)
        if m > 0 or n >= 0
    ]
    TH, VV = np.meshgrid(th, vv)
    rc, zs = [], []
    for m, n in modes:
        a = m * TH - n * VV
        wgt = 1.0 if (m == 0 and n == 0) else 2.0
        rc.append(wgt * np.mean(R * np.cos(a)))
        zs.append(wgt * np.mean(Z * np.sin(a)))
    return modes, np.array(rc), np.array(zs)


def _half(c_in, c_out, s_a, s_b, s_h, odd):
    """VMEC's parity-aware half-point value and slope of one row pair."""
    ev, ed = 0.5 * (c_in + c_out), (c_out - c_in) / (s_b - s_a)
    qa, qb = c_in / np.sqrt(s_a), c_out / np.sqrt(s_b)
    ov = np.sqrt(s_h) * 0.5 * (qa + qb)
    od = np.sqrt(s_h) * (qb - qa) / (s_b - s_a) + ov / (2.0 * s_h)
    return np.where(odd, ov, ev), np.where(odd, od, ed)


def _hermite(ya, da, yb, db, sa, sb, s):
    H = sb - sa
    t = (s - sa) / H
    sec = (yb - ya) / H
    al, be = da - sec, db - sec
    h10, h11 = t**3 - 2 * t**2 + t, t**3 - t**2
    g10, g11 = 3 * t**2 - 4 * t + 1, 3 * t**2 - 2 * t
    return ya + t * (yb - ya) + H * (h10 * al + h11 * be), sec + g10 * al + g11 * be


def resonant_harmonic(
    w: Surfaces, M: int = 24, N: int = 20, m: int = M_RES, n: int = N_RES, nu: int = 128
):
    """The torus integral of sqrt(g*) cos(m theta* - n v) at the crossing, the
    crossing radius, and mu0 dp/ds there from the wout's half-grid pressure."""
    j, s_star = rational_node(w, m, n)
    rows = [pest_rows(w, jj, M, N) for jj in (j - 1, j, j + 1)]
    modes = rows[0][0]
    mm = np.array([a for a, _ in modes], float)
    nn = np.array([b for _, b in modes], float)
    odd = (mm % 2) == 1
    sa, sj, sb = w.s_full[j - 1], w.s_full[j], w.s_full[j + 1]
    shm, shp = w.s_half[j], w.s_half[j + 1]
    R = [r[1] for r in rows]
    Z = [r[2] for r in rows]
    cRm, dRm = _half(R[0], R[1], sa, sj, shm, odd)
    cRp, dRp = _half(R[1], R[2], sj, sb, shp, odd)
    cZm, dZm = _half(Z[0], Z[1], sa, sj, shm, odd)
    cZp, dZp = _half(Z[1], Z[2], sj, sb, shp, odd)
    cR, cRs = _hermite(cRm, dRm, cRp, dRp, shm, shp, s_star)
    cZ, cZs = _hermite(cZm, dZm, cZp, dZp, shm, shp, s_star)
    nv = 4 * int(np.abs(nn).max() + abs(n) + 1)
    th = 2 * np.pi * np.arange(nu) / nu
    vv = 2 * np.pi * np.arange(nv) / nv
    TH, VV = np.meshgrid(th, vv)
    ang = mm[:, None, None] * TH[None] - nn[:, None, None] * VV[None]
    c, sn = np.cos(ang), np.sin(ang)
    Rv = np.tensordot(cR, c, 1)
    Ru = np.tensordot(-mm * cR, sn, 1)
    Rs = np.tensordot(cRs, c, 1)
    Zu = np.tensordot(mm * cZ, c, 1)
    Zs = np.tensordot(cZs, sn, 1)
    sg = Rv * (Ru * Zs - Rs * Zu)
    G = float(np.mean(sg * np.cos(m * TH - n * VV))) * 4 * np.pi**2
    return G, s_star, j


def mu0_dpds(wout, j: int) -> float:
    """mu0 dp/ds between the two half points of node j, from the wout's
    half-grid pressure in Pascal; exact for the linear profile of this case."""
    pres = np.asarray(wout.pres, dtype=float)
    h = 1.0 / (int(wout.ns) - 1)
    return MU_0 * (pres[j + 1] - pres[j]) / h


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ns", type=int, nargs="+", default=[49])
    ap.add_argument("--pres-scale", type=float, default=1.0)
    ap.add_argument("--M", type=int, default=24)
    ap.add_argument("--N", type=int, default=20)
    a = ap.parse_args()
    print(f"{'ns':>5} {'crossing s':>12} {'mu0 dp/ds':>12} {'G_11,10':>14}")
    for ns in a.ns:
        wout = run_w7x(ns, a.pres_scale)
        G, s_star, j = resonant_harmonic(Surfaces(wout), a.M, a.N)
        print(f"{ns:5d} {s_star:12.6f} {mu0_dpds(wout, j):12.4e} {G:+14.6e}")


if __name__ == "__main__":
    main()
