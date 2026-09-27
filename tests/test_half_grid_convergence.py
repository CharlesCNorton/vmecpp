# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Second-order convergence of the half-grid rule against certified enclosures.

`examples/half_grid_convergence.py` solves the half-grid rule's collocated system
for three manufactured mappings on the annulus 1/4 < s < 1: a three-dimensional
one, the same with non-stellarator-symmetric content, and that at five times the
pressure. The distance max |x_h - x*| of the discrete solution x_h from the
mapping's own coefficients x* is the discretization error.

Stellarocq (https://github.com/CharlesCNorton/stellarocq, gen/mms_colloc.py)
establishes each discrete solution by the interval Newton test on the collocated
system (Colloc.colloc_correct): exactly one zero lies in a box three units of a
58-bit mantissa wide in each unknown, around a centre refined by Newton steps on
the system's outputs read at 128 bits (Newton.centre_tab, Wide.v). That box gives
the error to the digits below. The enclosures fall by 3.81, 3.92 and 3.97 per
doubling of ns for the three-dimensional mapping and by 3.73 and 3.87 for the
others, the second order HalfGrid.lax_second_order asks of the discretization, and
the pressure, which enters the radial force only as a flux function the rule and
its source share, leaves the error unchanged. The floating-point solve is held to
the enclosures up to its own accuracy, a relative 1e-9.
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import half_grid_convergence as hgc

# case: {ns: (lower, upper)} enclosing max |x_h - x*|
CERTIFIED = {
    "3d": {
        9: (2.7343293671611875e-06, 2.734329367161228e-06),
        17: (7.169121696713442e-07, 7.169121696713848e-07),
        33: (1.8303258473727804e-07, 1.830325847373187e-07),
        65: (4.607481830039875e-08, 4.6074818300439406e-08),
    },
    "asymmetric": {
        9: (2.6907305700613703e-05, 2.6907305700613723e-05),
        17: (7.2066705290445185e-06, 7.206670529044539e-06),
        33: (1.8626687727518211e-06, 1.8626687727518415e-06),
    },
    "high-beta": {
        9: (2.6907305700613496e-05, 2.6907305700613517e-05),
        17: (7.206670529044363e-06, 7.206670529044383e-06),
        33: (1.8626687727516958e-06, 1.862668772751716e-06),
    },
}
REL = 1e-9

# the floating-point solves this file repeats, the fine grids being slow in Python
PAIRS = [(name, ns) for name, table in CERTIFIED.items() for ns in table if ns <= 17]


@pytest.fixture(scope="module")
def errors():
    cases = hgc.cases()
    out = {}
    for name, ns in PAIRS:
        pb = hgc.Problem(cases[name], ns)
        x, _ = pb.solve()
        out[name, ns] = float(np.abs(x - pb.xstar).max())
    return out


@pytest.mark.parametrize(("name", "ns"), PAIRS)
def test_error_within_the_certificate(errors, name, ns):
    lo, hi = CERTIFIED[name][ns]
    err = errors[name, ns]
    assert lo * (1 - REL) <= err <= hi * (1 + REL), (name, ns, err, lo, hi)


@pytest.mark.parametrize("name", list(CERTIFIED))
def test_certified_error_falls_fourfold(name):
    """Between consecutive resolutions the ratio of the errors, over every value the two
    enclosures allow, lies between 3.5 and 4.5."""
    table = CERTIFIED[name]
    ns = sorted(table)
    for a, b in pairwise(ns):
        assert table[a][0] / table[b][1] > 3.5, (name, a, b)
        assert table[a][1] / table[b][0] < 4.5, (name, a, b)


def test_pressure_leaves_the_error_unchanged():
    for ns in CERTIFIED["high-beta"]:
        a, b = CERTIFIED["asymmetric"][ns], CERTIFIED["high-beta"][ns]
        assert abs(a[0] - b[0]) <= 1e-12 * a[0], ns
