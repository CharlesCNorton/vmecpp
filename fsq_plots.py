"""Invariant force residual against iteration, plain descent versus Anderson
acceleration, for the QUASR configurations (fixed and free boundary).

    fsq_plots.py <input_dir> <out_dir> [fixed|free]

Runs every quasr*_<mode>.json of the input directory twice through vmecpp.run
(8 threads), with `anderson_acceleration` off and on, and writes one grid figure
per mode plus a CSV of the residual histories."""

import csv
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import vmecpp

input_dir, out_dir = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "fixed"
os.makedirs(out_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(input_dir, f"quasr*_{mode}.json")))


def residual_history(path, accelerate):
    vmec_input = vmecpp.VmecInput.from_file(path)
    vmec_input.anderson_acceleration = accelerate
    vmec_input.return_outputs_even_if_not_converged = True
    status = "ok"
    try:
        out = vmecpp.run(vmec_input, max_threads=8, verbose=False)
    except RuntimeError as exc:
        return None, str(exc).splitlines()[0][:60]
    w = out.wout
    fsq = np.asarray(w.force_residual_r) + np.asarray(w.force_residual_z) + np.asarray(w.force_residual_lambda)
    if w.ier_flag != 0:
        status = f"ier_flag {w.ier_flag}"
    return fsq, status


histories = {}
for path in files:
    case = os.path.basename(path)[:-5]
    for accelerate in (False, True):
        fsq, status = residual_history(path, accelerate)
        histories[(case, accelerate)] = (fsq, status)
        n = 0 if fsq is None else len(fsq)
        print(f"{case} accelerated={accelerate}: {n} iterations, {status}", flush=True)

with open(os.path.join(out_dir, f"fsq_{mode}.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case", "accelerated", "iteration", "fsq_invariant"])
    for (case, accelerate), (fsq, status) in histories.items():
        if fsq is None:
            continue
        for i, value in enumerate(fsq):
            writer.writerow([case, int(accelerate), i + 1, f"{value:.6e}"])

n_cases = len(files)
cols = 4
rows = (n_cases + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
for k, path in enumerate(files):
    case = os.path.basename(path)[:-5]
    ax = axes[k // cols][k % cols]
    for accelerate, color, label in ((False, "#1f77b4", "plain"), (True, "#d62728", "Anderson")):
        fsq, status = histories[(case, accelerate)]
        if fsq is None:
            ax.plot([], [], color=color, label=f"{label}: {status}")
            continue
        suffix = "" if status == "ok" else f" ({status})"
        ax.semilogy(np.arange(1, len(fsq) + 1), fsq, color=color, lw=0.9, label=f"{label}, {len(fsq)} it{suffix}")
    ax.set_title(case.replace("quasr", "QUASR ").replace("_", " "), fontsize=10)
    ax.set_xlabel("iteration")
    ax.set_ylabel("fsqr + fsqz + fsql")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
for k in range(n_cases, rows * cols):
    axes[k // cols][k % cols].axis("off")
fig.suptitle(f"QUASR {mode} boundary, ns [8, 16, 31], mpol = ntor = 6, ftol 1e-9: plain descent vs Anderson acceleration")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, f"fsq_{mode}.png"), dpi=110)
print("wrote", os.path.join(out_dir, f"fsq_{mode}.png"))
