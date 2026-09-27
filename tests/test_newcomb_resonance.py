# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""W7-X at finite beta against Newcomb's condition on its iota = 10/11 surface.

`examples/newcomb_resonance.py` runs the W7-X case of the test data at ns = 49,
finds the crossing of iota through 10/11, and integrates the Jacobian of
straight-field-line angles against cos(11 theta* - 10 v) over the angles there.
A smooth equilibrium with nested surfaces and p' != 0 on that surface has the
integral zero (Newcomb 1959).

The numbers below are enclosures established by Stellarocq
(https://github.com/CharlesCNorton/stellarocq, gen/newcomb.py; theorem
Newcomb.newcomb_violated) over every state whose straight-field-line R and Z
coefficients, truncated to m <= 24 and |n| <= 20 nfp, lie within a relative
1e-8 of those of this run: the integral lies in CERTIFIED_G, iota crosses 10/11
inside the interval over which the integral is taken, and |mu0 dp/ds| is at
least CERTIFIED_MU0_DPDS there. So no state of that box is a smooth
nested-surface equilibrium, and the resonant drive |mu0 p' G|, the product that
sets the Pfirsch-Schlueter current the surface would have to carry, is at least
CERTIFIED_DRIVE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import newcomb_resonance as nr

CERTIFIED_G = (-4.32857712346936950e-04, -2.67051724827280907e-04)
CERTIFIED_MU0_DPDS = 2.08550824036203813e-01
CERTIFIED_DRIVE = CERTIFIED_MU0_DPDS * 2.67051724827280907e-04


@pytest.fixture(scope="module")
def ns49():
    wout = nr.run_w7x(49)
    G, s_star, j = nr.resonant_harmonic(nr.Surfaces(wout), 24, 20)
    return wout, G, s_star, j


def test_resonant_harmonic_lies_in_the_certified_enclosure(ns49):
    _, G, _, _ = ns49
    lo, hi = CERTIFIED_G
    assert lo <= G <= hi, (G, lo, hi)


def test_pressure_gradient_holds_the_certified_floor(ns49):
    wout, _, _, j = ns49
    assert abs(nr.mu0_dpds(wout, j)) >= CERTIFIED_MU0_DPDS


def test_resonant_drive_holds_the_certified_floor(ns49):
    wout, G, _, j = ns49
    assert abs(nr.mu0_dpds(wout, j) * G) >= CERTIFIED_DRIVE


def test_crossing_sits_between_the_node_half_points(ns49):
    wout, _, s_star, j = ns49
    w = nr.Surfaces(wout)
    assert w.s_half[j] <= s_star <= w.s_half[j + 1]
