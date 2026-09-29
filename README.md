# lbsubs figures for proximafusion/vmecpp#537

`write_inputs.py`, `run_vmecpp.py`, `run_parvmec.sh`, `plots.py` and `sensitivity.py` make
the figures in this directory. With `VMECPP_SRC` a checkout of the PR branch, VMEC++
installed from it, `LBSUBS_OUT` the directory for the runs and figures, and `XVMEC` the
PARVMEC executable:

```shell
python write_inputs.py   # VMEC++ JSON and PARVMEC namelists, lbsubs off and on
python run_vmecpp.py     # VMEC++ runs, jxbout fields to vmecpp_{F,T}.npz
bash run_parvmec.sh      # PARVMEC runs, jxbout_<case>_{F,T}.nc
python plots.py          # figures in $LBSUBS_OUT/figs
python sensitivity.py    # W7-X at ftol 1e-14 against 1e-13
```

`write_inputs.py` converts the inputs with the branch's own `vmecpp_json_to_indata`, loaded
from `src/vmecpp/_util.py`.
