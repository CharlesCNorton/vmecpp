# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""W7-X vacuum flux surfaces from VMEC++ against certified tori of the coil field.

The coil set is simsopt's W7-X standard configuration (the five non-planar coil
types at 1.62 MA, the planar coils off). Stellarocq
(https://github.com/CharlesCNorton/stellarocq, gen/w7x_torus.py and
gen/coil_torus.py; theories/Coil.v) computes tori of that field from single
field lines by weighted Birkhoff averages and certifies, at every point of a
grid on each torus, a bound on the sine of the angle between the field and the
torus (Coil.coil_grid_bound); under the a posteriori KAM theorem, carried as
Hypotheses.kam_from_points, an invariant torus lies close to each
(Coil.coil_torus_from_grid). A torus is a flux surface around the magnetic axis
or a torus of the 5/5 island chain around its O-point, each a Fourier series

    R = sum Rc cos(m theta - n phi),   Z = sum Zs sin(m theta - n phi)

in the angle theta that conjugates the field lines to a rotation, which for a
flux surface is the straight-field-line angle.

This file writes an mgrid of the coil field with simsopt, runs VMEC++ in free
boundary in vacuum against it, and reads its surfaces the same way:
`surface_series` returns the Fourier coefficients of the VMEC++ surface through
a given point of the outboard midplane, in the straight-field-line angle
theta* = u + lambda, and `inside` decides whether points of a cross-section lie
inside a torus's cross-section at the same toroidal angle.

Usage:
    python w7x_vacuum_tori.py
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
from simsopt.configs import get_w7x_data
from simsopt.field import BiotSavart, coils_via_symmetries

import vmecpp

DATA = Path(__file__).resolve().parent / "data"
NFP = 5


def coil_field():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        curves, currents, _ = get_w7x_data()
    return BiotSavart(coils_via_symmetries(curves, currents, NFP, True))


def run_vacuum(ns: int = 49, phiedge: float = -1.74, workdir: Path | None = None):
    """VMEC++ in free boundary, no pressure and no current, against an mgrid of the coil
    field written by simsopt."""
    workdir = Path(workdir or tempfile.mkdtemp())
    mgrid = workdir / "mgrid_w7x_simsopt.nc"
    if not mgrid.exists():
        coil_field().to_mgrid(
            str(mgrid),
            nr=96,
            nz=96,
            nphi=36,
            rmin=4.3,
            rmax=6.7,
            zmin=-1.2,
            zmax=1.2,
            nfp=NFP,
        )
    vi = vmecpp.VmecInput.from_file(DATA / "w7x.json")
    steps = [x for x in (13, 25) if x < ns] + [ns]
    vi = vi.model_copy(
        update={
            "lfreeb": True,
            "mgrid_file": str(mgrid),
            "extcur": np.array([1.0]),
            "nvacskip": 6,
            "ncurr": 1,
            "curtor": 0.0,
            "am": np.array([0.0]),
            "ac": np.array([0.0]),
            "pres_scale": 0.0,
            "phiedge": phiedge,
            "ns_array": np.array(steps),
            "ftol_array": np.full(len(steps), 1e-13),
            "niter_array": np.full(len(steps), 30000),
        },
        deep=True,
    )
    return vmecpp.run(vi, verbose=False).wout


def _rows(wout, name):
    a = np.asarray(getattr(wout, name), dtype=float)
    return a.T if a.shape[0] == len(np.asarray(wout.xm)) else a


def surface_series(wout, R_out: float, modes, nth: int = 96, nph: int = 48):
    """Rc and Zs, in the straight-field-line angle over one field period, of the VMEC++
    surface through the point (R_out, 0) of the outboard midplane at phi = 0, and the
    radius s of that surface."""
    xm = np.asarray(wout.xm, float)
    xn = np.asarray(wout.xn, float)
    R, Z, L = _rows(wout, "rmnc"), _rows(wout, "zmns"), _rows(wout, "lmns")
    ns = R.shape[0]
    s_full = np.linspace(0.0, 1.0, ns)
    s_half = 0.5 * (s_full[1:] + s_full[:-1])
    # u = 0 at phi = 0 is the outboard midplane, where theta* = u
    ss = np.linspace(0.0, 1.0, 20001)
    Rmid = np.array([np.interp(ss, s_full, R[:, i]) for i in range(len(xm))]).sum(
        axis=0
    )
    s = float(ss[np.argmin(np.abs(Rmid - R_out))])
    Rs = np.array([np.interp(s, s_full, R[:, i]) for i in range(len(xm))])
    Zs_ = np.array([np.interp(s, s_full, Z[:, i]) for i in range(len(xm))])
    Ls = np.array([np.interp(s, s_half, L[1:, i]) for i in range(len(xm))])
    # theta* = u + lambda(u, v): invert on a fine grid of u for each phi
    ths = 2 * np.pi * np.arange(nth) / nth
    phs = (2 * np.pi / NFP) * np.arange(nph) / nph
    u = np.linspace(0, 2 * np.pi, 4 * nth, endpoint=False)
    RR = np.zeros((nph, nth))
    ZZ = np.zeros((nph, nth))
    for j, ph in enumerate(phs):
        tu = u + Ls @ np.sin(xm[:, None] * u[None] - xn[:, None] * ph)
        uu = np.interp(ths, tu, u, period=2 * np.pi)
        for _ in range(4):  # Newton on theta*(u) = theta
            a2 = xm[:, None] * uu[None] - xn[:, None] * ph
            f = uu + Ls @ np.sin(a2) - ths
            df = 1 + (xm * Ls) @ np.cos(a2)
            uu = uu - f / df
        a2 = xm[:, None] * uu[None] - xn[:, None] * ph
        RR[j] = Rs @ np.cos(a2)
        ZZ[j] = Zs_ @ np.sin(a2)
    CR = np.fft.fft2(RR) / (nph * nth)
    CZ = np.fft.fft2(ZZ) / (nph * nth)
    Rc, Zc = [], []
    for m, n in modes:
        w = 1.0 if (m == 0 and n == 0) else 2.0
        cr = CR[(-(n // NFP)) % nph, m % nth]
        cz = CZ[(-(n // NFP)) % nph, m % nth]
        Rc.append(w * cr.real)
        Zc.append(-w * cz.imag)
    return np.array(Rc), np.array(Zc), float(s)


def torus_section(torus: dict, phi: float, nth: int = 256):
    """R and Z of a certified torus's cross-section at the toroidal angle phi."""
    m = np.array([a for a, _ in torus["modes"]], float)
    n = np.array([b for _, b in torus["modes"]], float)
    th = 2 * np.pi * np.arange(nth) / nth
    ang = m[:, None] * th[None] - n[:, None] * phi
    return np.asarray(torus["Rc"]) @ np.cos(ang), np.asarray(torus["Zs"]) @ np.sin(ang)


def inside(torus: dict, phi: float, R, Z):
    """Whether points (R, Z) at the toroidal angle phi lie inside the torus's cross-
    section, by the winding number of the section around each point."""
    Rt, Zt = torus_section(torus, phi)
    R = np.atleast_1d(R)
    Z = np.atleast_1d(Z)
    ang = np.arctan2(Zt[None, :] - Z[:, None], Rt[None, :] - R[:, None])
    d = np.diff(np.concatenate([ang, ang[:, :1]], axis=1), axis=1)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return np.abs(d.sum(axis=1)) > np.pi


def main():
    w = run_vacuum()
    io = np.asarray(w.iotaf)
    print(
        f"VMEC++ vacuum W7-X: ns {w.ns}, iota {io[0]:.5f} .. {io[-1]:.5f}, "
        f"volume {w.volume_p:.3f} m^3"
    )


if __name__ == "__main__":
    main()
