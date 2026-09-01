"""Side-by-side animation of the flux surfaces during the first multigrid stage,
plain descent versus Anderson acceleration.

    surface_animation.py <input.json> <out_gif> <max_frames> [stride]

Each frame is the state after k iterations, obtained by running the stage with
niter_array = [k] and return_outputs_even_if_not_converged; the descent is
deterministic, so the truncated runs trace the same path. Frames are every
`stride` iterations, or geometrically spaced when stride is 0. Cross-sections
at zeta = 0 and zeta = pi / nfp."""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

import vmecpp

path, out_gif = sys.argv[1], sys.argv[2]
max_frames = int(sys.argv[3])
stride = int(sys.argv[4]) if len(sys.argv) > 4 else 1
case = os.path.basename(path)[:-5]

base = vmecpp.VmecInput.from_file(path)
base.ns_array = np.array(base.ns_array[:1], dtype=np.int64)
base.ftol_array = np.array(base.ftol_array[:1])
base.niter_array = np.array(base.niter_array[:1], dtype=np.int64)
base.return_outputs_even_if_not_converged = True


def full_run(accelerate):
    inp = base.model_copy(update={"anderson_acceleration": accelerate})
    try:
        out = vmecpp.run(inp, max_threads=8, verbose=False)
        w = out.wout
        fsq = np.asarray(w.force_residual_r) + np.asarray(w.force_residual_z) + np.asarray(w.force_residual_lambda)
        return len(fsq), fsq, "converged" if w.ier_flag == 0 else f"ier_flag {w.ier_flag}"
    except RuntimeError as exc:
        return None, None, str(exc).splitlines()[0][:50]


def state_after(accelerate, k):
    inp = base.model_copy(update={"anderson_acceleration": accelerate, "niter_array": np.array([k], dtype=np.int64)})
    try:
        out = vmecpp.run(inp, max_threads=8, verbose=False)
    except RuntimeError:
        return None
    return out.wout


def cross_sections(w, zeta):
    xm, xn = np.asarray(w.xm), np.asarray(w.xn)
    theta = np.linspace(0.0, 2.0 * np.pi, 121)
    rmnc, zmns = np.asarray(w.rmnc), np.asarray(w.zmns)  # (mnmax, ns)
    angle = np.outer(xm, theta) - np.outer(xn, np.full_like(theta, zeta))  # (mnmax, ntheta)
    r = rmnc.T @ np.cos(angle)  # (ns, ntheta)
    z = zmns.T @ np.sin(angle)
    return r, z


lengths = {}
traces = {}
status = {}
failed_at = {}
for accelerate in (False, True):
    lengths[accelerate], traces[accelerate], status[accelerate] = full_run(accelerate)
    print(f"{case} accelerated={accelerate}: {lengths[accelerate]} iterations, {status[accelerate]}", flush=True)
    if lengths[accelerate] is None:
        # a run that aborts: find how far truncated runs still return a state
        lo, hi = 1, int(base.niter_array[0])
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if state_after(accelerate, mid) is not None:
                lo = mid
            else:
                hi = mid - 1
        lengths[accelerate] = lo
        failed_at[accelerate] = lo
        print(f"  truncated runs return a state up to iteration {lo}", flush=True)
n_max = max(lengths.values())
if stride > 0:
    frames = list(range(1, min(n_max, max_frames * stride) + 1, stride))
    if frames[-1] != min(n_max, max_frames * stride):
        frames.append(min(n_max, max_frames * stride))
else:
    ratio = np.exp(np.log(n_max) / (max_frames - 1))
    frames = sorted({1, n_max} | {int(round(ratio**i)) for i in range(max_frames)})
    frames = [f for f in frames if 1 <= f <= n_max]

states = {}  # cache of (accelerate, k) -> wout, so the final state is computed once
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
writer = PillowWriter(fps=8)
nfp = int(base.nfp)
with writer.saving(fig, out_gif, dpi=72):
    for k in frames:
        for col, accelerate in enumerate((False, True)):
            k_eff = min(k, lengths[accelerate])
            if (accelerate, k_eff) not in states:
                states[(accelerate, k_eff)] = state_after(accelerate, k_eff)
            w = states[(accelerate, k_eff)]
            for row, zeta in enumerate((0.0, np.pi / nfp)):
                ax = axes[row][col]
                ax.clear()
                if w is not None:
                    r, z = cross_sections(w, zeta)
                    for j in range(r.shape[0]):
                        ax.plot(r[j], z[j], color="#1f77b4" if not accelerate else "#d62728", lw=0.8)
                    ax.plot(r[-1], z[-1], color="k", lw=1.2)
                ax.set_aspect("equal")
                fsq_now = traces[accelerate][k_eff - 1] if traces[accelerate] is not None and k_eff <= len(traces[accelerate]) else float("nan")
                label = "plain" if not accelerate else "Anderson"
                tag = f"iteration {k_eff}"
                if accelerate in failed_at and k_eff >= failed_at[accelerate]:
                    tag += " (last state before abort)"
                ax.set_title(f"{label}, {tag}, fsq {fsq_now:.1e}, zeta = {'0' if zeta == 0 else 'pi/nfp'}", fontsize=9)
                ax.set_xlabel("R [m]")
                ax.set_ylabel("Z [m]")
                ax.grid(alpha=0.3)
        fig.suptitle(f"{case}: first multigrid stage (ns = {int(base.ns_array[0])}); plain {lengths[False]} it ({status[False]}), Anderson {lengths[True]} it ({status[True]})", fontsize=10)
        fig.tight_layout()
        writer.grab_frame()
        print(f"frame {k}", flush=True)
print("wrote", out_gif)
