#!/bin/bash
# Run PARVMEC on the inputs write_inputs.py wrote, lbsubs off and on, one case per
# argument (default: every case directory under $LBSUBS_OUT).
X=${XVMEC:-xvmec}
OUT=${LBSUBS_OUT:-lbsubs_plots}
cases="$*"
[ -z "$cases" ] && cases=$(ls "$OUT")
for c in $cases; do
  for tag in F T; do
    d="$OUT/$c"
    [ -f "$d/input.${c}_$tag" ] || { echo "$c $tag: no input"; continue; }
    start=$(date +%s)
    (cd "$d" && mpirun -np 1 "$X" "input.${c}_$tag" > "parvmec_$tag.log" 2>&1)
    rc=$?
    printf '%s %s rc=%s secs=%s jxbout=%s\n' "$c" "$tag" "$rc" "$(( $(date +%s) - start ))" \
      "$(ls "$d/jxbout_${c}_$tag.nc" 2>/dev/null | wc -l)"
  done
done
