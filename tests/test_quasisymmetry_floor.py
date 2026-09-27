# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The precise quasisymmetric configurations of Landreman and Paul (2021) against
certified floors on their quasisymmetry defect.

`examples/quasisymmetry_floor.py` computes, on the half point outside a node, the
defect D = min over lam of max |t1 - lam t2| of the two-term quasisymmetry criterion
over 32 x 16 angles of one field period. Stellarocq
(https://github.com/CharlesCNorton/stellarocq, gen/qs_floor.py with --rel 1e-8;
theorem QSFloor.qs_floor) proves that every state whose R and Z on the node and its
two neighbours, lambda on the two half points around it and iota there lie within a
relative 1e-8 of the run's, coefficient by coefficient, has for every lam a point
with |t1 - lam t2| >= the floor:

- the reactor-scale quasi-helically symmetric configuration at node 25 of 51, where
  the harmonics of the two terms against sin(u + 12 v) and sin(2 u + 4 v) have
  ratios whose certified enclosures, [-5.5625, -5.5280] and [-5.6432, -5.6098],
  are apart;
- the quasi-axisymmetric configuration at node 100 of 201, where against sin(u)
  and sin(3 u + 6 v) they are [-192.27, -190.09] and [-120.84, -14.89].

A run of either input lands in its box, since VMEC++ reproduces it exactly. A run
from a boundary perturbed by as little as a relative 1e-12 does not: the solver's
spread in the small lambda coefficients, a few 1e-10 absolute, exceeds a relative
1e-8 of them. Such runs of the quasi-helical configuration are checked against its
floor as a regression, which the certificate itself does not cover.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import quasisymmetry_floor as qf

CERTIFIED_FLOOR = {
    "QH": 7.70937186248149665e04,
    "QA": 4.89461953854493928e-12,
}
REL = 1e-8


def _rows(w, j):
    R, Z, L = qf._rows(w, "rmnc"), qf._rows(w, "zmns"), qf._rows(w, "lmns")
    iota = np.asarray(w.iotas, dtype=float)[j : j + 2]
    return [R[j - 1], R[j], R[j + 1], Z[j - 1], Z[j], Z[j + 1], L[j], L[j + 1], iota]


def _in_box(rows, rows0):
    """Every coefficient within half the certified relative width of the certified
    run's own, a zero one within a quarter of its row's largest."""
    for r, r0 in zip(rows, rows0, strict=True):
        big = float(np.abs(r0).max())
        tol = np.where(r0 != 0.0, 0.5 * REL * np.abs(r0), 0.25 * REL * big)
        if np.any(np.abs(r - r0) > tol):
            return False
    return True


@pytest.fixture(scope="module")
def baseline():
    return {case: qf.run(case=case) for case in qf.CASES}


@pytest.mark.parametrize("case", list(qf.CASES))
def test_defect_holds_the_certified_floor(baseline, case):
    j = qf.CASES[case][1]
    D, _ = qf.best_defect(*qf.terms(baseline[case], j=j))
    assert CERTIFIED_FLOOR[case] <= D, (case, D)


@pytest.mark.parametrize("case", list(qf.CASES))
def test_rerun_lands_in_the_certified_box(baseline, case):
    j = qf.CASES[case][1]
    assert _in_box(_rows(qf.run(case=case), j), _rows(baseline[case], j))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_perturbed_boundary_stays_above_the_floor(seed):
    j = qf.CASES["QH"][1]
    w = qf.run(perturb=1e-10, seed=seed, case="QH")
    D, _ = qf.best_defect(*qf.terms(w, j=j))
    assert CERTIFIED_FLOOR["QH"] <= D, (seed, D)
