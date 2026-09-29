"""Figures comparing VMEC++'s lbsubs with PARVMEC's, from the outputs of run_vmecpp.py
and run_parvmec.sh."""

import os
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

OUT = Path(os.environ.get("LBSUBS_OUT", "lbsubs_plots"))
FIGS = OUT / "figs"
TITLES = {
    "solovev": "Solov'ev",
    "cth_like_fixed_bdy": "CTH-like",
    "cma": "CMA",
    "w7x": "W7-X",
}
CASE_COLORS = {
    "solovev": "#1baf7a",
    "cth_like_fixed_bdy": "#2a78d6",
    "cma": "#9b59b6",
    "w7x": "#eb6834",
}
VMECPP = "#2a78d6"
PARVMEC = "#eb6834"
OFF = "#898781"
INK = "#0b0b0b"
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 110,
}
RESIDUAL_LABEL = r"$(J\times B)_s - p'$, relative"


def scaled(field):
    """The field in units of a power of ten, and that power."""
    top = float(np.max(np.abs(field)))
    power = int(np.floor(np.log10(top))) if top > 0 else 0
    return field / 10.0**power, power


def load_parvmec(case, tag):
    """PARVMEC's jxbout, surface arrays transposed to (ns, nzeta, ntheta)."""
    with netCDF4.Dataset(OUT / case / f"jxbout_{case}_{tag}.nc") as d:
        out = {name: np.asarray(v[...]) for name, v in d.variables.items()}
    ns = int(out["radial_surfaces"])
    for name, value in list(out.items()):
        if value.ndim == 3:
            out[name] = np.transpose(value, (2, 1, 0))
            assert out[name].shape[0] == ns, (name, value.shape)
    out["bsubs3"] = out["bsubs"]
    return out


def load_vmecpp(case, tag, shape):
    """VMEC++'s jxbout, surface arrays reshaped to (ns, nzeta, ntheta)."""
    _, nzeta, ntheta = shape
    raw = dict(np.load(OUT / case / f"vmecpp_{tag}.npz"))
    for name, value in list(raw.items()):
        if value.ndim == 2 and value.shape[1] == nzeta * ntheta:
            raw[name] = value.reshape(value.shape[0], nzeta, ntheta)
    return raw


def load(case):
    data = {}
    for tag in ("F", "T"):
        p = data[("P", tag)] = load_parvmec(case, tag)
        data[("V", tag)] = load_vmecpp(case, tag, p["bsubs"].shape)
    return data


def grids(data):
    ns, nzeta, ntheta = data[("P", "F")]["bsubs"].shape
    nfp = int(data[("V", "F")]["nfp"])
    theta = np.linspace(0.0, np.pi, ntheta)
    zeta = 2.0 * np.pi / nfp * np.arange(nzeta) / nzeta
    s = np.linspace(0.0, 1.0, ns)
    return s, theta, zeta, nfp


def inner(ns):
    return slice(1, ns - 1)


def mid(ns):
    return int(round(0.5 * (ns - 1)))


def surface_scale(data, code):
    """The largest |(J x B)_s - p'| of each surface with lbsubs = F."""
    return np.max(np.abs(data[(code, "F")]["jxb_gradp"]), axis=(1, 2))


def spread(data, code, tag):
    """The spread of (J x B)_s - p' over each surface, relative to surface_scale."""
    r = data[(code, tag)]["jxb_gradp"]
    scale = surface_scale(data, code)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.max(r, axis=(1, 2)) - np.min(r, axis=(1, 2))) / scale


def surface_map(ax, field, theta, zeta, nfp, norm):
    """A (theta, zeta) map of one surface over one field period."""
    period = 360.0 / nfp
    z = np.append(np.degrees(zeta), period)
    values = np.concatenate([field, field[:1]], axis=0)
    mesh = ax.pcolormesh(
        np.degrees(theta), z, values, shading="nearest", cmap="RdBu_r", norm=norm
    )
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_xticks([0, 90, 180])
    return mesh


def symmetric_norm(*fields):
    top = max(float(np.max(np.abs(f))) for f in fields)
    return mpl.colors.TwoSlopeNorm(vmin=-top, vcenter=0.0, vmax=top)


def title(fig, text):
    fig.suptitle(text, x=0.07, ha="left", fontsize=11, fontweight="bold")


def spread_panel(ax, data, s, j=None):
    ns = len(s)
    for code, color, style, name in (
        ("V", VMECPP, "-", "VMEC++"),
        ("P", PARVMEC, "--", "PARVMEC"),
    ):
        for tag, shade, width in (("F", OFF, 1.0), ("T", color, 1.5)):
            ax.semilogy(
                s[inner(ns)],
                spread(data, code, tag)[inner(ns)],
                color=shade,
                ls=style,
                lw=width,
                label=f"{name}, lbsubs = {tag}",
            )
    if j is not None:
        ax.axvline(s[j], color=OFF, lw=0.6, ls=":")
    ax.set_xlabel("s")
    ax.set_ylabel("(max - min) over the surface, relative")
    ax.set_title("spread of the residual on every surface", loc="left")
    ax.legend(fontsize=8)


def fig_force_residual(case, data):
    """The local radial force residual on a mid-radius surface, lbsubs off and on,
    and its spread on every surface."""
    s, theta, zeta, nfp = grids(data)
    j = mid(len(s))
    scale = {code: surface_scale(data, code)[j] for code in ("V", "P")}
    residual = {
        key: data[key]["jxb_gradp"][j] / scale[key[0]]
        for key in (("V", "F"), ("V", "T"), ("P", "T"))
    }
    fig = plt.figure(figsize=(13.4, 3.9))
    grid = fig.add_gridspec(
        1, 6, width_ratios=[1, 1, 1, 0.05, 0.85, 1.6], wspace=0.12
    )
    norm = symmetric_norm(residual[("V", "F")])
    for k, (key, label) in enumerate(
        (
            (("V", "F"), "VMEC++, lbsubs = F"),
            (("V", "T"), "VMEC++, lbsubs = T"),
            (("P", "T"), "PARVMEC, lbsubs = T"),
        )
    ):
        ax = fig.add_subplot(grid[0, k])
        mesh = surface_map(ax, residual[key], theta, zeta, nfp, norm)
        ax.set_title(label, loc="left")
        if k == 0:
            ax.set_ylabel(r"$\zeta$ (deg)")
        else:
            ax.tick_params(labelleft=False)
    fig.colorbar(mesh, cax=fig.add_subplot(grid[0, 3]), label=RESIDUAL_LABEL)
    spread_panel(fig.add_subplot(grid[0, 5]), data, s, j)
    title(fig, f"{TITLES[case]}: local radial force residual, maps at s = {s[j]:.2f}")
    return fig


def fig_bsubs_correction(case, data):
    """The lbsubs correction to B_s on a mid-radius surface in both codes, their
    difference, and the correction's size on every surface."""
    s, theta, zeta, nfp = grids(data)
    ns = len(s)
    j = mid(ns)
    dv = data[("V", "T")]["bsubs3"] - data[("V", "F")]["bsubs3"]
    dp = data[("P", "T")]["bsubs3"] - data[("P", "F")]["bsubs3"]
    fig = plt.figure(figsize=(13.4, 3.9))
    grid = fig.add_gridspec(
        1, 8, width_ratios=[1, 1, 0.05, 0.4, 1, 0.05, 0.55, 1.5], wspace=0.12
    )
    _, power = scaled(np.concatenate([dv[j].ravel(), dp[j].ravel()]))
    norm = symmetric_norm(dv[j] / 10.0**power, dp[j] / 10.0**power)
    for k, (field, label) in enumerate(((dv[j], "VMEC++"), (dp[j], "PARVMEC"))):
        ax = fig.add_subplot(grid[0, k])
        mesh = surface_map(ax, field / 10.0**power, theta, zeta, nfp, norm)
        ax.set_title(label, loc="left")
        if k == 0:
            ax.set_ylabel(r"$\zeta$ (deg)")
        else:
            ax.tick_params(labelleft=False)
    fig.colorbar(
        mesh, cax=fig.add_subplot(grid[0, 2]), label=rf"$\Delta B_s$ ($10^{{{power}}}$ T)"
    )
    ax = fig.add_subplot(grid[0, 4])
    diff, dpower = scaled(dv[j] - dp[j])
    mesh = surface_map(ax, diff, theta, zeta, nfp, symmetric_norm(diff))
    ax.set_title("VMEC++ - PARVMEC", loc="left")
    fig.colorbar(
        mesh, cax=fig.add_subplot(grid[0, 5]), label=rf"$10^{{{dpower}}}$ T"
    )

    ax = fig.add_subplot(grid[0, 7])
    rows = inner(ns)
    base = np.max(np.abs(data[("V", "F")]["bsubs3"][rows]), axis=(1, 2))
    ax.semilogy(s[rows], base, color=OFF, lw=1.0, label=r"$|B_s|$, lbsubs = F")
    ax.semilogy(
        s[rows], np.max(np.abs(dv[rows]), axis=(1, 2)), color=VMECPP,
        label=r"$|\Delta B_s|$, VMEC++",
    )
    ax.semilogy(
        s[rows], np.max(np.abs(dp[rows]), axis=(1, 2)), color=PARVMEC, ls="--",
        label=r"$|\Delta B_s|$, PARVMEC",
    )
    ax.semilogy(
        s[rows], np.max(np.abs(dv[rows] - dp[rows]), axis=(1, 2)), color=INK, lw=1.0,
        label=r"$|\Delta B_s|$, VMEC++ - PARVMEC",
    )
    ax.axvline(s[j], color=OFF, lw=0.6, ls=":")
    ax.set_xlabel("s")
    ax.set_ylabel("max over the surface (T)")
    ax.set_title("on every surface", loc="left")
    ax.legend(fontsize=8)
    title(
        fig,
        f"{TITLES[case]}: "
        r"$\Delta B_s = B_s(\mathrm{lbsubs=T}) - B_s(\mathrm{lbsubs=F})$, maps at "
        f"s = {s[j]:.2f}",
    )
    return fig


def fig_axisymmetric(case, data):
    """B_s and the local radial force residual against theta on three surfaces, and
    the spread of the residual on every surface."""
    s, theta, _, _ = grids(data)
    ns = len(s)
    rows = [int(round(x * (ns - 1))) for x in (0.25, 0.5, 0.75)]
    shades = ("#86b6ef", "#2a78d6", "#184f95")
    theta_deg = np.degrees(theta)
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.9), gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    for j, shade in zip(rows, shades, strict=True):
        ax.plot(
            theta_deg, data[("V", "T")]["bsubs3"][j, 0], color=shade, lw=1.5,
            label=f"s = {s[j]:.2f}",
        )
        ax.plot(
            theta_deg, data[("P", "T")]["bsubs3"][j, 0], "o", ms=3.4, mfc="none",
            mec=PARVMEC, mew=0.9,
        )
    ax.plot([], [], "o", ms=3.4, mfc="none", mec=PARVMEC, label="PARVMEC")
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel(r"$B_s$ (T)")
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_title(r"$B_s$, lbsubs = T; lines VMEC++", loc="left")
    ax.legend(fontsize=8)

    ax = axes[1]
    correction = {
        code: data[(code, "T")]["bsubs3"] - data[(code, "F")]["bsubs3"] for code in "VP"
    }
    _, power = scaled(correction["V"][rows])
    for j, shade in zip(rows, shades, strict=True):
        ax.plot(theta_deg, correction["V"][j, 0] / 10.0**power, color=shade, lw=1.5,
                label=f"s = {s[j]:.2f}")
        ax.plot(theta_deg, correction["P"][j, 0] / 10.0**power, "o", ms=3.4,
                mfc="none", mec=PARVMEC, mew=0.9)
    ax.plot([], [], "o", ms=3.4, mfc="none", mec=PARVMEC, label="PARVMEC")
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel(rf"$\Delta B_s$ ($10^{{{power}}}$ T)")
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_title(r"$\Delta B_s = B_s(\mathrm{T}) - B_s(\mathrm{F})$; lines VMEC++",
                 loc="left")
    ax.legend(fontsize=8)

    ax = axes[2]
    for code in ("V", "P"):
        scale = surface_scale(data, code)
        for j, shade in zip(rows, shades, strict=True):
            if code == "V":
                ax.plot(theta_deg, data[("V", "F")]["jxb_gradp"][j, 0] / scale[j],
                        color=shade, ls="--", lw=1.0)
                ax.plot(theta_deg, data[("V", "T")]["jxb_gradp"][j, 0] / scale[j],
                        color=shade, lw=1.5, label=f"s = {s[j]:.2f}")
            else:
                ax.plot(theta_deg, data[("P", "T")]["jxb_gradp"][j, 0] / scale[j], "o",
                        ms=3.4, mfc="none", mec=PARVMEC, mew=0.9)
    ax.plot([], [], color=OFF, ls="--", label="lbsubs = F")
    ax.plot([], [], color=OFF, label="lbsubs = T")
    ax.plot([], [], "o", ms=3.4, mfc="none", mec=PARVMEC, label="PARVMEC, lbsubs = T")
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel(RESIDUAL_LABEL)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_title("local radial force residual; lines VMEC++", loc="left")
    ax.legend(fontsize=8)

    spread_panel(axes[3], data, s)
    title(fig, f"{TITLES[case]}: " + r"$B_s$ and the local radial force residual")
    return fig


def fig_summary(cases, datasets):
    """Across the cases: the spread of the local radial force residual on every
    surface, and the agreement of B_s between the codes."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3), gridspec_kw={"wspace": 0.28})
    for case in cases:
        data = datasets[case]
        s = grids(data)[0]
        rows = inner(len(s))
        color = CASE_COLORS[case]
        for tag, style, width in (("F", "--", 1.0), ("T", "-", 1.5)):
            axes[0].semilogy(
                s[rows], spread(data, "V", tag)[rows], color=color, ls=style, lw=width,
                label=f"{TITLES[case]}, lbsubs = {tag}",
            )
        dv = data[("V", "T")]["bsubs3"] - data[("V", "F")]["bsubs3"]
        dp = data[("P", "T")]["bsubs3"] - data[("P", "F")]["bsubs3"]
        axes[1].semilogy(
            s[rows],
            np.max(np.abs(dv[rows] - dp[rows]), axis=(1, 2)) / np.max(np.abs(dp[rows])),
            color=color, lw=1.5, label=rf"{TITLES[case]}: $\Delta B_s$",
        )
        base_v = data[("V", "F")]["bsubs3"][rows]
        base_p = data[("P", "F")]["bsubs3"][rows]
        axes[1].semilogy(
            s[rows],
            np.max(np.abs(base_v - base_p), axis=(1, 2)) / np.max(np.abs(base_p)),
            color=color, lw=1.0, ls=":",
            label=rf"{TITLES[case]}: $B_s$, lbsubs = F",
        )
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("(max - min) over the surface, relative")
    axes[0].set_title("VMEC++: spread of the local radial force residual", loc="left")
    axes[0].legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("max |VMEC++ - PARVMEC| / max |PARVMEC|")
    axes[1].set_title(r"VMEC++ against PARVMEC: $B_s$ and $\Delta B_s$", loc="left")
    axes[1].legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    return fig


def describe(case, data):
    s = grids(data)[0]
    ns = len(s)
    rows = inner(ns)
    j = mid(ns)
    dv = data[("V", "T")]["bsubs3"] - data[("V", "F")]["bsubs3"]
    dp = data[("P", "T")]["bsubs3"] - data[("P", "F")]["bsubs3"]
    base_v, base_p = data[("V", "F")]["bsubs3"][rows], data[("P", "F")]["bsubs3"][rows]
    print(
        f"{case}: ns={ns} shape={data[('P', 'F')]['bsubs'].shape} "
        f"max|B_s|={np.max(np.abs(base_p)):.3g} max|dB_s|={np.max(np.abs(dp[rows])):.3g}\n"
        f"  B_s(F) V vs P: {np.max(np.abs(base_v - base_p)) / np.max(np.abs(base_p)):.2e}"
        f"  dB_s V vs P: {np.max(np.abs(dv[rows] - dp[rows])) / np.max(np.abs(dp[rows])):.2e}\n"
        f"  spread at s={s[j]:.2f}: V F {spread(data, 'V', 'F')[j]:.3g} "
        f"V T {spread(data, 'V', 'T')[j]:.3g} P F {spread(data, 'P', 'F')[j]:.3g} "
        f"P T {spread(data, 'P', 'T')[j]:.3g}; max over s of T spread: "
        f"V {np.nanmax(spread(data, 'V', 'T')[rows]):.3g} "
        f"P {np.nanmax(spread(data, 'P', 'T')[rows]):.3g}"
    )


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    cases = sys.argv[1:] or [
        c
        for c in TITLES
        if (OUT / c / f"jxbout_{c}_T.nc").exists() and (OUT / c / "vmecpp_T.npz").exists()
    ]
    datasets = {}
    with mpl.rc_context(STYLE):
        for case in cases:
            data = datasets[case] = load(case)
            describe(case, data)
            if data[("P", "F")]["bsubs"].shape[1] == 1:
                makers = (("profiles", fig_axisymmetric),)
            else:
                makers = (
                    ("force_residual", fig_force_residual),
                    ("bsubs_correction", fig_bsubs_correction),
                )
            for name, make in makers:
                fig = make(case, data)
                fig.savefig(FIGS / f"{case}_{name}.png", bbox_inches="tight")
                plt.close(fig)
        fig = fig_summary(cases, datasets)
        fig.savefig(FIGS / "summary.png", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
