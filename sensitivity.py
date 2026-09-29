"""How much the W7-X lbsubs correction moves within each code between ftol 1e-14 and
1e-13, against the difference between the codes."""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import plots  # noqa: E402

cases = {"14": plots.load("w7x"), "13": plots.load("w7x_f13")}


def correction(data, code):
    return data[(code, "T")]["bsubs3"] - data[(code, "F")]["bsubs3"]


s = plots.grids(cases["14"])[0]
ns = len(s)
rows = plots.inner(ns)
d = {(code, f): correction(cases[f], code) for code in "VP" for f in cases}
scale = np.max(np.abs(d[("P", "14")][rows]))
curves = {
    "VMEC++ - PARVMEC, ftol 1e-14": d[("V", "14")] - d[("P", "14")],
    "VMEC++, ftol 1e-14 - 1e-13": d[("V", "14")] - d[("V", "13")],
    "PARVMEC, ftol 1e-14 - 1e-13": d[("P", "14")] - d[("P", "13")],
}
per_surface = {k: np.max(np.abs(v), axis=(1, 2)) / scale for k, v in curves.items()}
for j in (10, 30, 46, 47, 48, 60, 77, 78, 79, 80, 81, 90):
    print(
        f"j={j:3d} s={s[j]:.3f} "
        + "  ".join(f"{k}: {v[j]:.2e}" for k, v in per_surface.items())
    )
others = np.ones(ns, dtype=bool)
others[[0, 47, 78, 79, 80, ns - 1]] = False
for k, v in per_surface.items():
    print(f"{k}: max away from s = 0.48, 0.81: {np.max(v[others]):.2e}")

with mpl.rc_context(plots.STYLE):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    styles = {
        "VMEC++ - PARVMEC, ftol 1e-14": dict(color=plots.INK, lw=1.4),
        "VMEC++, ftol 1e-14 - 1e-13": dict(color=plots.VMECPP, lw=1.2),
        "PARVMEC, ftol 1e-14 - 1e-13": dict(color=plots.PARVMEC, lw=1.2, ls="--"),
    }
    for k, v in per_surface.items():
        ax.semilogy(s[rows], v[rows], label=k, **styles[k])
    ax.set_xlabel("s")
    ax.set_ylabel(r"max over the surface of the change in $\Delta B_s$, relative")
    ax.set_title(
        r"W7-X: $\Delta B_s$ between the codes and within each code", loc="left"
    )
    ax.legend(fontsize=8.5)
    fig.savefig(plots.FIGS / "w7x_sensitivity.png", bbox_inches="tight")
