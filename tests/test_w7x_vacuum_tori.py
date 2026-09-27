# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""W7-X vacuum flux surfaces from VMEC++ against certified tori of the coil field.

`tests/data/w7x_tori/certified_tori.json` holds tori of the W7-X standard
configuration's coil field (simsopt's coil set, the five non-planar coil types at
1.62 MA, the planar coils off), each computed from one field line by weighted
Birkhoff averages and written as

    R = sum Rc cos(m theta - n phi),   Z = sum Zs sin(m theta - n phi)

in the angle that conjugates the lines on it to a rotation. For every point of a
grid on each torus, Stellarocq (https://github.com/CharlesCNorton/stellarocq,
gen/coil_torus.py; theories/Coil.v, coil_grid_bound) certifies that the sine of the
angle between the field and the torus is at most `sine_bound`; under the a
posteriori KAM theorem, carried as Hypotheses.kam_from_points, an invariant torus of
the field lies close to each (Coil.coil_torus_from_grid). The flux surfaces carry a
transform `iota`; the island tori wind around the O-point of the 5/5 chain.

`examples/w7x_vacuum_tori.py` runs VMEC++ in free boundary in vacuum against the same
coil field. Its surface through a flux-surface torus's outboard midplane point,
written in the straight-field-line angle, matches the torus coefficient by
coefficient, and its boundary lies outside the cross-section of every certified
torus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("simsopt")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import w7x_vacuum_tori as wv

DATA = Path(__file__).resolve().parent / "data" / "w7x_tori" / "certified_tori.json"
NS = 49
# VMEC++ at NS surfaces against a certified torus, per Fourier coefficient
COEFFICIENT_TOL = 5e-4


@pytest.fixture(scope="module")
def tori():
    return json.loads(DATA.read_text())["tori"]


@pytest.fixture(scope="module")
def wout(tmp_path_factory):
    return wv.run_vacuum(ns=NS, workdir=tmp_path_factory.mktemp("w7x"))


def _modes(t):
    return [tuple(mn) for mn in t["modes"]]


def test_certified_tori_nearly_invariant(tori):
    """The float angle between simsopt's field and each torus, on the torus's own grid,
    stays within the certified bound."""
    bs = wv.coil_field()
    for t in tori:
        nu, nv = t["grid"]
        u = 2 * np.pi * np.arange(nu) / nu
        v = 2 * np.pi * np.arange(nv) / nv
        U, V = np.meshgrid(u, v)
        m = np.array([a for a, _ in t["modes"]], float)
        n = np.array([b for _, b in t["modes"]], float)
        ang = m[:, None] * U.ravel()[None] - n[:, None] * V.ravel()[None]
        C, S = np.cos(ang), np.sin(ang)
        Rc, Zs = np.asarray(t["Rc"]), np.asarray(t["Zs"])
        R, Ru, Rv = Rc @ C, (-m * Rc) @ S, (n * Rc) @ S
        Z, Zu, Zv = Zs @ S, (m * Zs) @ C, (-n * Zs) @ C
        cv, sv = np.cos(V.ravel()), np.sin(V.ravel())
        x = np.stack([R * cv, R * sv, Z], -1)
        xu = np.stack([Ru * cv, Ru * sv, Zu], -1)
        xv = np.stack([Rv * cv - R * sv, Rv * sv + R * cv, Zv], -1)
        nrm = np.cross(xu, xv)
        bs.set_points(x)
        B = bs.B()
        sine = np.sum(B * nrm, -1) / np.sqrt(np.sum(B * B, -1) * np.sum(nrm * nrm, -1))
        assert np.abs(sine).max() <= t["sine_bound"] * (1 + 1e-6), t["name"]


def test_vacuum_surface_matches_certified_torus(wout, tori):
    for t in tori:
        if t["kind"] != "surface":
            continue
        R_out = float(np.sum(t["Rc"]))
        Rc, Zs, s = wv.surface_series(wout, R_out, _modes(t), nth=96, nph=128)
        diff = max(
            np.abs(Rc - np.asarray(t["Rc"])).max(),
            np.abs(Zs - np.asarray(t["Zs"])).max(),
        )
        assert diff <= COEFFICIENT_TOL, (t["name"], s, diff)


def test_boundary_between_certified_tori(wout, tori):
    """At every point of the boundary on eight toroidal planes of a period, the boundary
    is outside every certified torus."""
    xm = np.asarray(wout.xm, float)
    xn = np.asarray(wout.xn, float)
    R = wv._rows(wout, "rmnc")[-1]
    Z = wv._rows(wout, "zmns")[-1]
    u = 2 * np.pi * np.arange(128) / 128
    for phi in 2 * np.pi / wv.NFP * np.arange(8) / 8:
        ang = xm[:, None] * u[None] - xn[:, None] * phi
        Rb, Zb = R @ np.cos(ang), Z @ np.sin(ang)
        for t in tori:
            assert not np.any(wv.inside(t, phi, Rb, Zb)), (t["name"], phi)
