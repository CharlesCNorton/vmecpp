"""Write each case's input, converged tightly, as VMEC++ JSON and as PARVMEC namelists
with lbsubs off and on, using the PR's own JSON-to-INDATA converter."""

import importlib.util
import json
import os
from pathlib import Path

# a checkout of the branch, and the directory the runs and figures go to
SRC = Path(os.environ.get("VMECPP_SRC", "."))
OUT = Path(os.environ.get("LBSUBS_OUT", "lbsubs_plots"))
TEST_DATA = SRC / "src/vmecpp/cpp/vmecpp/test_data"
# case: (source JSON, ftol of every multigrid stage)
CASES = {
    "solovev": (TEST_DATA / "solovev.json", 1.0e-16),
    "cth_like_fixed_bdy": (TEST_DATA / "cth_like_fixed_bdy.json", 1.0e-14),
    "cma": (TEST_DATA / "cma.json", 1.0e-14),
    "w7x": (SRC / "examples/data/w7x.json", 1.0e-14),
    # W7-X one decade less converged, for sensitivity.py
    "w7x_f13": (SRC / "examples/data/w7x.json", 1.0e-13),
}
NITER = 60000

spec = importlib.util.spec_from_file_location("pr_util", SRC / "src/vmecpp/_util.py")
util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util)

for name, (path, ftol) in CASES.items():
    case_dir = OUT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    entry = json.loads(path.read_text())
    stages = len(entry["ns_array"])
    entry["ftol_array"] = [ftol] * stages
    entry["niter_array"] = [NITER] * stages
    (case_dir / f"{name}.json").write_text(json.dumps(entry, indent=1))
    for flag in (False, True):
        entry["lbsubs"] = flag
        text = util.vmecpp_json_to_indata(entry)
        (case_dir / f"input.{name}_{'T' if flag else 'F'}").write_text(text)
    print(name, "ns", entry["ns_array"], "ftol", ftol, "nzeta", entry.get("nzeta"))
