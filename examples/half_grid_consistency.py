# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Consistency of the half-grid rule, against its theorem.

VMEC keeps R and Z on full-grid nodes and the field on the half points between
them. The radial force at a node is assembled from its two half points: a
quantity the force needs at the node is the average of its two half-point
values, and a radial derivative is their difference over the spacing h,

    r_s = (avg(d_v B_s) - dif(B_v)) avg(B^v) - (dif(B_u) - avg(d_u B_s)) avg(B^u)
          - mu0 dp/ds.

Stellarocq's theories/HalfGrid.v (https://github.com/CharlesCNorton/stellarocq)
proves that this differs from the exact radial force at the node by at most

    K h^2,   K = (|a| Mv/8 + |B^v| (Msv/8 + Mcv/24) + (Msv/8 + Mcv/24)(Mv/8) h^2)
               + (|b| Mu/8 + |B^u| (Mcu/24 + Msu/8) + (Mcu/24 + Msu/8)(Mu/8) h^2),

with a = d_v B_s - d_s B_v and b = d_s B_u - d_u B_s at the node, and M* bounds
on the second derivatives of the averaged quantities and the third derivatives
of the differenced ones over the interval between the half points
(node_second_order). This file evaluates both sides on the analytic mappings of
`manufactured_solution.py`, at the node and on a grid of angles: the rule's
error, and the constant the theorem charges.

The field of a mapping follows the solver's conventions: L = phip lambda,
sqrt(g) B^u = chip - dL/dv and sqrt(g) B^v = phip + dL/du, with v the
geometric toroidal angle. Radial derivatives are taken by complex step, so the
exact force is exact to rounding.
"""

from __future__ import annotations

import manufactured_solution as mms
import numpy as np


class Field:
    """The six radial profiles the rule reads, at fixed angles, as functions of a
    real or complex radius."""

    def __init__(self, model: mms.Model, u: np.ndarray, v: np.ndarray):
        self.model = model
        self.u = u
        self.w = v * model.case.nfp  # the per-period angle the mapping is written in
        c = model.case
        self.phip = float(c.phip)
        self.iota = c.iota_coeff
        self.am = c.am
        self.pres_scale = c.pres_scale

    def _q(self, name, key, s):
        # the model's v-derivatives are taken in the geometric angle
        fn = self.model.fn[name + "|" + key]
        return fn(s, self.u, self.w) * np.ones_like(self.u * self.w)

    def chip(self, s):
        return self.phip * sum(ci * s**i for i, ci in enumerate(self.iota))

    def mu0_dpds(self, s):
        return (
            mms.MU_0
            * self.pres_scale
            * sum(i * ci * s ** (i - 1) for i, ci in enumerate(self.am) if i > 0)
        )

    def quantities(self, s):
        """B^u, B^v, B_u, B_v, d_u B_s, d_v B_s at radius s."""
        q = self._q
        R, Rs, Ru, Rv = q("R", "", s), q("R", "s", s), q("R", "u", s), q("R", "v", s)
        Zs, Zu, Zv = q("Z", "s", s), q("Z", "u", s), q("Z", "v", s)
        Rsu, Rsv = q("R", "su", s), q("R", "sv", s)
        Ruu, Ruv, Rvv = q("R", "uu", s), q("R", "uv", s), q("R", "vv", s)
        Zsu, Zsv = q("Z", "su", s), q("Z", "sv", s)
        Zuu, Zuv, Zvv = q("Z", "uu", s), q("Z", "uv", s), q("Z", "vv", s)
        Lu, Lv = q("L", "u", s), q("L", "v", s)
        Luu, Luv, Lvv = q("L", "uu", s), q("L", "uv", s), q("L", "vv", s)
        tau = Ru * Zs - Rs * Zu
        g = R * tau
        tau_u = Ruu * Zs + Ru * Zsu - Rsu * Zu - Rs * Zuu
        tau_v = Ruv * Zs + Ru * Zsv - Rsv * Zu - Rs * Zuv
        g_u = Ru * tau + R * tau_u
        g_v = Rv * tau + R * tau_v
        guu = Ru**2 + Zu**2
        guv = Ru * Rv + Zu * Zv
        gvv = Rv**2 + Zv**2 + R**2
        gsu = Rs * Ru + Zs * Zu
        gsv = Rs * Rv + Zs * Zv
        gsu_u = Rsu * Ru + Rs * Ruu + Zsu * Zu + Zs * Zuu
        gsu_v = Rsv * Ru + Rs * Ruv + Zsv * Zu + Zs * Zuv
        gsv_u = Rsu * Rv + Rs * Ruv + Zsu * Zv + Zs * Zuv
        gsv_v = Rsv * Rv + Rs * Rvv + Zsv * Zv + Zs * Zvv
        nu_ = self.chip(s) - Lv
        nv_ = self.phip + Lu
        Bu = nu_ / g
        Bv = nv_ / g
        Bu_u = (-Luv * g - nu_ * g_u) / g**2
        Bv_u = (Luu * g - nv_ * g_u) / g**2
        Bu_v = (-Lvv * g - nu_ * g_v) / g**2
        Bv_v = (Luv * g - nv_ * g_v) / g**2
        return {
            "Bu": Bu,
            "Bv": Bv,
            "B_u": guu * Bu + guv * Bv,
            "B_v": guv * Bu + gvv * Bv,
            "B_s_u": gsu_u * Bu + gsu * Bu_u + gsv_u * Bv + gsv * Bv_u,
            "B_s_v": gsu_v * Bu + gsu * Bu_v + gsv_v * Bv + gsv * Bv_v,
        }

    def dds(self, s, step=1e-30):
        """The radial derivative of every quantity, by complex step."""
        z = self.quantities(s + 1j * step)
        return {k: np.imag(val) / step for k, val in z.items()}


def exact_rs(field: Field, s: float) -> np.ndarray:
    q = {k: np.real(v) for k, v in field.quantities(complex(s)).items()}
    d = field.dds(s)
    return (
        (q["B_s_v"] - d["B_v"]) * q["Bv"]
        - (d["B_u"] - q["B_s_u"]) * q["Bu"]
        - field.mu0_dpds(s)
    )


def node_rs(field: Field, s: float, h: float) -> np.ndarray:
    """The half-grid rule, fed the exact half-point values at s -+ h/2."""
    qm = {k: np.real(v) for k, v in field.quantities(complex(s - h / 2)).items()}
    qp = {k: np.real(v) for k, v in field.quantities(complex(s + h / 2)).items()}

    def avg(k):
        return 0.5 * (qm[k] + qp[k])

    def dif(k):
        return (qp[k] - qm[k]) / h

    return (
        (avg("B_s_v") - dif("B_v")) * avg("Bv")
        - (dif("B_u") - avg("B_s_u")) * avg("Bu")
        - field.mu0_dpds(s)
    )


def theorem_constant(field: Field, s: float, h: float, nsample: int = 17) -> np.ndarray:
    """K of node_second_order at every angle, with the derivative bounds read as
    the largest magnitude over nsample radii of the interval."""
    xs = np.linspace(s - h / 2, s + h / 2, nsample)
    delta = h / 64.0
    second = dict.fromkeys(("Bu", "Bv", "B_s_u", "B_s_v"), 0.0)
    third = dict.fromkeys(("B_u", "B_v"), 0.0)
    for x in xs:
        d0 = field.dds(x)
        dp = field.dds(x + delta)
        dm = field.dds(x - delta)
        for k in second:
            second[k] = np.maximum(second[k], np.abs((dp[k] - dm[k]) / (2 * delta)))
        for k in third:
            third[k] = np.maximum(
                third[k], np.abs((dp[k] - 2 * d0[k] + dm[k]) / delta**2)
            )
    q = {k: np.real(v) for k, v in field.quantities(complex(s)).items()}
    d = field.dds(s)
    a = np.abs(q["B_s_v"] - d["B_v"])
    b = np.abs(d["B_u"] - q["B_s_u"])
    Mu, Mv = second["Bu"], second["Bv"]
    Msu, Msv = second["B_s_u"], second["B_s_v"]
    Mcu, Mcv = third["B_u"], third["B_v"]
    e1 = Msv / 8 + Mcv / 24
    e2 = Mcu / 24 + Msu / 8
    return (a * Mv / 8 + np.abs(q["Bv"]) * e1 + e1 * (Mv / 8) * h * h) + (
        b * Mu / 8 + np.abs(q["Bu"]) * e2 + e2 * (Mu / 8) * h * h
    )


def cases():
    """Three mappings: the fitted three-dimensional one, the same with
    non-stellarator-symmetric content, and that at five times the pressure."""
    sym = mms.build_case(mms.FITTED_P)
    asym = mms.build_case(mms.FITTED_P, asym=mms.ASYM)
    high = mms.build_case(
        mms.FITTED_P, base=dict(mms.FITTED_BASE, pres_scale=5 * 160000.0), asym=mms.ASYM
    )
    return {"3d": sym, "asymmetric": asym, "high-beta": high}


def study(s=0.5, ns_list=(17, 33, 65, 129), nu=12, nv=8):
    u, v = np.meshgrid(
        2 * np.pi * np.arange(nu) / nu, 2 * np.pi * np.arange(nv) / nv, indexing="ij"
    )
    for name, case in cases().items():
        field = Field(mms.Model(case), u, v)
        exact = exact_rs(field, s)
        prev = None
        print(f"{name}: max |r_s| at the node {np.max(np.abs(exact)):.3e}")
        for ns in ns_list:
            h = 1.0 / (ns - 1)
            err = np.abs(node_rs(field, s, h) - exact)
            k = theorem_constant(field, s, h)
            ratio = "" if prev is None else f"  fall x{prev / np.max(err):.2f}"
            print(
                f"  ns={ns:4d} max error {np.max(err):.3e}  "
                f"max error / (K h^2) {np.max(err / (k * h * h)):.3f}{ratio}"
            )
            prev = np.max(err)


if __name__ == "__main__":
    study()
