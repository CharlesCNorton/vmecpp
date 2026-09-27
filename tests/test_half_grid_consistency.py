# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The half-grid rule against its consistency theorem, and the solver's order at
high beta.

`examples/half_grid_consistency.py` evaluates the radial force at a full-grid
node by VMEC's half-grid rule, from exact half-point values of an analytic
mapping, and compares it with the exact force there. Stellarocq's
theories/HalfGrid.v proves the difference is at most K h^2 with K built from the
mapping's radial derivatives (node_second_order); these tests take that K as the
tolerance, on a three-dimensional mapping, the same with
non-stellarator-symmetric content, and that at five times the pressure.
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sympy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import half_grid_consistency as hgc
import manufactured_solution as mms

S_NODE = 0.5
NS = (17, 33, 65)
MPOL = 4
NTOR = 2
NTHETA = 18
NZETA = 16
SMIN = 0.2


@pytest.fixture(scope="module")
def fields():
    u, v = np.meshgrid(
        2 * np.pi * np.arange(12) / 12, 2 * np.pi * np.arange(8) / 8, indexing="ij"
    )
    return {
        name: hgc.Field(mms.Model(case), u, v) for name, case in hgc.cases().items()
    }


@pytest.mark.parametrize("name", ["3d", "asymmetric", "high-beta"])
def test_node_rule_is_within_the_theorem_bound(fields, name):
    field = fields[name]
    exact = hgc.exact_rs(field, S_NODE)
    for ns in NS:
        h = 1.0 / (ns - 1)
        err = np.abs(hgc.node_rs(field, S_NODE, h) - exact)
        bound = hgc.theorem_constant(field, S_NODE, h, nsample=33) * h * h
        assert np.all(err <= bound), (name, ns, float(np.max(err / bound)))


@pytest.mark.parametrize("name", ["3d", "asymmetric", "high-beta"])
def test_node_rule_falls_fourfold(fields, name):
    field = fields[name]
    exact = hgc.exact_rs(field, S_NODE)
    errors = [
        float(np.max(np.abs(hgc.node_rs(field, S_NODE, 1.0 / (ns - 1)) - exact)))
        for ns in NS
    ]
    for coarse, fine in pairwise(errors):
        assert 3.9 < coarse / fine < 4.1, errors


def test_high_beta_asymmetric_force_is_second_order():
    """The solver's discrete force converges to the continuum force at second order
    on the three-dimensional non-stellarator-symmetric mapping at five times the
    fitted pressure."""
    case = mms.build_case(
        mms.FITTED_P, base=dict(mms.FITTED_BASE, pres_scale=5 * 160000.0), asym=mms.ASYM
    )
    model = mms.Model(case)
    errors = []
    for ns in (13, 25, 49):
        fv, _ = mms.discrete_force(case, ns, MPOL, NTOR, NTHETA, NZETA, lasym=True)
        sgrid = np.linspace(0.0, 1.0, ns)
        fa = mms.hat_force_to_decomposed(
            mms.project_force(model, case, sgrid, MPOL, NTOR, nu=32, nw=32, lasym=True),
            MPOL,
        )
        errors.append(max(mms._force_by_parity(fv, fa, ns, MPOL, SMIN, True)))
    assert errors == sorted(errors, reverse=True), errors
    assert np.log(errors[-2] / errors[-1]) / np.log(2.0) > 1.7, errors
