# ruff: noqa
"""Throwaway probe: the OpenMP runtimes of the wheel-test process, and what happens when an
idle vmecpp worker thread goes to sleep. Each variant runs in its own interpreter."""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "examples" / "data" / "solovev.json"

COMMON = r'''
import ctypes, faulthandler, sys, time
faulthandler.enable()

def omp_images():
    libc = ctypes.CDLL(None)
    libc._dyld_image_count.restype = ctypes.c_uint32
    libc._dyld_get_image_name.restype = ctypes.c_char_p
    libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    names = [libc._dyld_get_image_name(i).decode() for i in range(libc._dyld_image_count())]
    return [n for n in names if "omp" in n.rsplit("/", 1)[-1].lower()]
'''

SIMSOPT_FIRST = "from simsopt.geo import SurfaceRZFourier\nimport vmecpp\n"
VMECPP_FIRST = "import vmecpp\nfrom simsopt.geo import SurfaceRZFourier\n"


def body(thread_counts):
    lines = ['print("OpenMP images in load order:", omp_images(), flush=True)',
             'inp = vmecpp.VmecInput.from_file(r"%s")' % INPUT]
    for n in thread_counts:
        lines.append("out = vmecpp.run(inp, max_threads=%r, verbose=False)" % n)
        lines.append('print("run with max_threads=%r done after", out.wout.niter, "iterations", flush=True)' % n)
    lines.append("time.sleep(1.5)  # longer than KMP_BLOCKTIME, so idle workers go to sleep")
    lines.append('print("SURVIVED", flush=True)')
    return "\n".join(lines) + "\n"


VARIANTS = [
    ("A: simsopt imported first, run on 3 threads", SIMSOPT_FIRST, [3]),
    ("B: vmecpp imported first, run on 3 threads", VMECPP_FIRST, [3]),
    ("C: simsopt imported first, run on 1 thread", SIMSOPT_FIRST, [1]),
    ("E: simsopt first, run with max_threads=1, then a run with the default", SIMSOPT_FIRST, [1, None]),
]

print("python", sys.version.split()[0], "cpu_count", os.cpu_count(), "OpenMP_ROOT=%r" % os.environ.get("OpenMP_ROOT"), flush=True)
so = subprocess.run([sys.executable, "-c", "import simsoptpp; print(simsoptpp.__file__)"], capture_output=True, text=True).stdout.strip()
print("simsoptpp:", so, flush=True)
print(subprocess.run(["otool", "-L", so], capture_output=True, text=True).stdout, flush=True)

results = []
for title, imports, thread_counts in VARIANTS:
    print("=" * 100, flush=True)
    print(title, flush=True)
    proc = subprocess.run([sys.executable, "-c", COMMON + imports + body(thread_counts)], capture_output=True, text=True)
    print(proc.stdout, flush=True)
    print(proc.stderr[-3500:], flush=True)
    results.append((title, proc.returncode))

print("=" * 100, flush=True)
for title, code in results:
    print("PROBE RESULT  exit %4d  %s" % (code, title), flush=True)
