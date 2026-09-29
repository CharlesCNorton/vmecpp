"""Run VMEC++ with lbsubs off and on on each case's input from write_inputs.py and save
the jxbout fields."""

import os
import sys
from pathlib import Path

import numpy as np

import vmecpp

OUT = Path(os.environ.get("LBSUBS_OUT", "lbsubs_plots"))
CASES = ("solovev", "cth_like_fixed_bdy", "cma", "w7x", "w7x_f13")
FIELDS = (
    "bsubs3",
    "jxb_gradp",
    "jcrossb",
    "itheta",
    "izeta",
    "jdotb_sqrtg",
    "sqrtg3",
    "bsupu3",
    "bsupv3",
    "amaxfor",
    "aminfor",
    "avforce",
    "pprim",
    "phin",
    "jdotb",
)

names = sys.argv[1:] or list(CASES)
for name in names:
    case_dir = OUT / name
    path = case_dir / f"{name}.json"
    for flag in (False, True):
        tag = "T" if flag else "F"
        vmec_input = vmecpp.VmecInput.from_file(path).model_copy(update={"lbsubs": flag})
        output = vmecpp.run(vmec_input, verbose=False, max_threads=4)
        arrays = {f: np.asarray(getattr(output.jxbout, f)) for f in FIELDS}
        wout = output.wout
        arrays.update(
            ns=wout.ns,
            nfp=wout.nfp,
            mpol=wout.mpol,
            ntor=wout.ntor,
            iotaf=np.asarray(wout.iotaf),
            ier_flag=wout.ier_flag,
            fsqr=wout.fsqr,
            fsqz=wout.fsqz,
            fsql=wout.fsql,
            nzeta_input=vmec_input.nzeta,
            ntheta_input=vmec_input.ntheta,
        )
        np.savez(case_dir / f"vmecpp_{tag}.npz", **arrays)
        print(
            f"{name} lbsubs={tag}: ns={wout.ns} ier={wout.ier_flag} "
            f"fsq=({wout.fsqr:.1e},{wout.fsqz:.1e},{wout.fsql:.1e}) "
            f"bsubs3 {arrays['bsubs3'].shape} amaxfor[ns//2]={arrays['amaxfor'][wout.ns // 2]:.3g} "
            f"aminfor[ns//2]={arrays['aminfor'][wout.ns // 2]:.3g}",
            flush=True,
        )
