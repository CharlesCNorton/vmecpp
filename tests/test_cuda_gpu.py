# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Regression tests for the CUDA execution path.

These require a CUDA-enabled build (-DVMECPP_USE_CUDA=ON) and an NVIDIA GPU, so they are
opt-in: set VMECPP_TEST_CUDA=1 to run them. The W7-X case additionally requires
VMECPP_TEST_CUDA_SLOW=1 (several minutes of GPU time).
"""

import os
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import vmecpp
from vmecpp.cpp import _vmecpp

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"
W7X_JSON = REPO_ROOT / "examples" / "data" / "w7x.json"

pytestmark = pytest.mark.skipif(
    os.environ.get("VMECPP_TEST_CUDA") != "1",
    reason="CUDA GPU tests are opt-in via VMECPP_TEST_CUDA=1",
)

# Converged cma references, validated bit-identical across sm_80, sm_86,
# and sm_89 (see the PR validation battery). The distinct-mode values
# correspond to the unscaled boundary and the 1.02-scaled boundary.
CMA_VOLUME = 0.501352745718
CMA_ASPECT = 6.107316546103
CMA_VOLUME_SCALED = 0.532039544578


def _cma_input():
    return vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cma.json")


def _single_env():
    os.environ["VMECPP_N_CONFIG_MAX"] = "1"
    for key in ("VMECPP_BATCH_DISTINCT", "VMECPP_PER_CFG_RECOMPUTE"):
        os.environ.pop(key, None)


def _distinct_pair():
    vi = _cma_input()
    vi2 = vi.model_copy(deep=True)
    vi2.rbc = vi2.rbc * 1.02
    vi2.zbs = vi2.zbs * 1.02
    return [vi._to_cpp_vmecindata(), vi2._to_cpp_vmecindata()]


def test_single_cma_canonical():
    _single_env()
    out = vmecpp.run(_cma_input(), max_threads=1, verbose=False)
    assert out.wout.volume == pytest.approx(CMA_VOLUME, rel=1e-11)
    assert out.wout.aspect == pytest.approx(CMA_ASPECT, rel=1e-11)


def test_repeated_run_bit_identical():
    # The same input run twice in one process returns bitwise-identical
    # results; the persistent device state carries nothing between runs.
    _single_env()
    first = vmecpp.run(_cma_input(), max_threads=1, verbose=False)
    second = vmecpp.run(_cma_input(), max_threads=1, verbose=False)
    assert second.wout.volume == first.wout.volume
    assert second.wout.aspect == first.wout.aspect
    assert second.wout.wb == first.wout.wb


def test_broadcast_returns_single_output():
    # Broadcast mode solves one boundary in every slot and returns the
    # single converged result. This path also exercises the iteration-1
    # magnetic-axis recovery, whose re-staged device state must reach
    # every configuration slot.
    _single_env()
    vi = _cma_input()
    outs = _vmecpp.run_batched_gpu(
        [vi._to_cpp_vmecindata()] * 4, max_threads=1, verbose=_vmecpp.OutputMode.SILENT
    )
    assert len(outs) == 1
    assert outs[0].wout.volume == pytest.approx(CMA_VOLUME, rel=1e-11)


def test_two_call_distinct_identical():
    # Two consecutive distinct-mode calls in one process must return
    # bitwise-identical converged spectra: the second call restages its
    # inputs from clean device state.
    _single_env()
    os.environ["VMECPP_BATCH_DISTINCT"] = "1"
    try:
        inds = _distinct_pair()
        outs1, sp1 = _vmecpp.run_batched_gpu(
            inds, max_threads=1, verbose=_vmecpp.OutputMode.SILENT, return_spectra=True
        )
        outs2, sp2 = _vmecpp.run_batched_gpu(
            _distinct_pair(),
            max_threads=1,
            verbose=_vmecpp.OutputMode.SILENT,
            return_spectra=True,
        )
    finally:
        os.environ.pop("VMECPP_BATCH_DISTINCT", None)
    assert len(outs1) == 2
    assert outs1[0].wout.volume == pytest.approx(CMA_VOLUME, rel=1e-11)
    assert outs1[1].wout.volume == pytest.approx(CMA_VOLUME_SCALED, rel=1e-11)
    assert np.array_equal(sp1, sp2)
    assert outs2[0].wout.volume == outs1[0].wout.volume
    assert outs2[1].wout.volume == outs1[1].wout.volume


def test_mixed_configuration_counts_one_process():
    # distinct N=2, broadcast N=4, distinct N=2, single: the
    # per-configuration buffers reallocate across the count changes and
    # every run returns its fresh-process result.
    _single_env()
    os.environ["VMECPP_BATCH_DISTINCT"] = "1"
    outs = _vmecpp.run_batched_gpu(
        _distinct_pair(), max_threads=1, verbose=_vmecpp.OutputMode.SILENT
    )
    assert outs[1].wout.volume == pytest.approx(CMA_VOLUME_SCALED, rel=1e-11)

    os.environ.pop("VMECPP_BATCH_DISTINCT", None)
    vi = _cma_input()
    outs = _vmecpp.run_batched_gpu(
        [vi._to_cpp_vmecindata()] * 4, max_threads=1, verbose=_vmecpp.OutputMode.SILENT
    )
    assert len(outs) == 1
    assert outs[0].wout.volume == pytest.approx(CMA_VOLUME, rel=1e-11)

    os.environ["VMECPP_BATCH_DISTINCT"] = "1"
    try:
        outs = _vmecpp.run_batched_gpu(
            _distinct_pair(), max_threads=1, verbose=_vmecpp.OutputMode.SILENT
        )
    finally:
        os.environ.pop("VMECPP_BATCH_DISTINCT", None)
    assert outs[0].wout.volume == pytest.approx(CMA_VOLUME, rel=1e-11)
    assert outs[1].wout.volume == pytest.approx(CMA_VOLUME_SCALED, rel=1e-11)

    _single_env()
    out = vmecpp.run(_cma_input(), max_threads=1, verbose=False)
    assert out.wout.volume == pytest.approx(CMA_VOLUME, rel=1e-11)


def test_scope_guards_and_memory_preflight():
    _single_env()
    vi = _cma_input()

    vi_theta = vi.model_copy(deep=True)
    vi_theta.ntheta = 70
    with pytest.raises(Exception, match="nThetaReduced"):
        vmecpp.run(vi_theta, max_threads=1, verbose=False)

    vi_ns = vi.model_copy(deep=True)
    vi_ns.ns_array = np.array([1500])
    vi_ns.ftol_array = np.array([1e-6])
    vi_ns.niter_array = np.array([10])
    with pytest.raises(Exception, match="1024"):
        vmecpp.run(vi_ns, max_threads=1, verbose=False)

    os.environ["VMECPP_N_CONFIG_MAX"] = "20000"
    try:
        with pytest.raises(Exception, match="free memory"):
            vmecpp.run(vi, max_threads=1, verbose=False)
    finally:
        os.environ["VMECPP_N_CONFIG_MAX"] = "1"

    # The same process recovers and runs normally after the rejections.
    out = vmecpp.run(vi, max_threads=1, verbose=False)
    assert out.wout.volume == pytest.approx(CMA_VOLUME, rel=1e-11)


def test_sync_elision_and_iteration_graph_bit_identical():
    # K-window sync elision converges cma to the canonical result, and
    # the whole-iteration CUDA graph replays the elided iterations to
    # bitwise the same converged state.
    _single_env()
    vi = _cma_input()
    os.environ["VMECPP_SYNC_ELIDE"] = "25"
    try:
        out_elide = vmecpp.run(vi, max_threads=1, verbose=False)
        os.environ["VMECPP_ITER_GRAPH"] = "1"
        out_graph = vmecpp.run(_cma_input(), max_threads=1, verbose=False)
    finally:
        os.environ.pop("VMECPP_SYNC_ELIDE", None)
        os.environ.pop("VMECPP_ITER_GRAPH", None)
    assert out_elide.wout.volume == pytest.approx(CMA_VOLUME, rel=1e-11)
    assert out_graph.wout.volume == out_elide.wout.volume
    assert out_graph.wout.aspect == out_elide.wout.aspect


def test_ozaki3_scatter_converges_in_family():
    # The 3-slice Ozaki FP32 scatter is the converging FP32-multiplication
    # path; it lands within a few ULP of the FP64 result.
    _single_env()
    os.environ["VMECPP_SCATTER_OZAKI3_FP32"] = "1"
    try:
        out = vmecpp.run(_cma_input(), max_threads=1, verbose=False)
    finally:
        os.environ.pop("VMECPP_SCATTER_OZAKI3_FP32", None)
    assert out.wout.volume == pytest.approx(CMA_VOLUME, rel=1e-12)
    assert out.wout.aspect == pytest.approx(CMA_ASPECT, rel=1e-12)


def test_i8gemm_staged_limb_descent_converges_in_family():
    # The batched int8-Ozaki scatter under the staged limb descent:
    # 4-limb operands above the IR residual threshold, 8 below, with
    # the width transition (and its whole-iteration-graph drop)
    # exercised mid-run at each multigrid stage.
    _single_env()
    os.environ["VMECPP_SCATTER_I8GEMM"] = "1"
    os.environ["VMECPP_IR_STAGED"] = "1"
    try:
        out = vmecpp.run(_cma_input(), max_threads=1, verbose=False)
    finally:
        os.environ.pop("VMECPP_SCATTER_I8GEMM", None)
        os.environ.pop("VMECPP_IR_STAGED", None)
    assert out.wout.volume == pytest.approx(CMA_VOLUME, rel=1e-12)
    assert out.wout.aspect == pytest.approx(CMA_ASPECT, rel=1e-12)


def test_free_boundary_multigrid_stage_transition():
    # The multigrid stage transition with the vacuum pressure already
    # active reallocates the device buffers mid-run; the constraint
    # origins enter the new stage zero-initialized because
    # rzConIntoVolume does not run there.
    _single_env()
    vi = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy_multigrid.json")
    vi.mgrid_file = str(TEST_DATA_DIR / "mgrid_cth_like.nc")
    out = vmecpp.run(vi, max_threads=1, verbose=False)
    assert out.wout.volume == pytest.approx(0.307720974219842, rel=1e-4)
    assert out.wout.aspect == pytest.approx(5.4351152498877, rel=1e-5)
    assert out.wout.wb == pytest.approx(0.00128356726624431, rel=1e-5)


def test_free_boundary_cth_like():
    # Free-boundary results carry the documented drift family rather
    # than the fixed-boundary bit-exact contract; the tolerances sit
    # just above the measured sm_80/sm_86 drift.
    _single_env()
    vi = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")
    vi.mgrid_file = str(TEST_DATA_DIR / "mgrid_cth_like.nc")
    out = vmecpp.run(vi, max_threads=1, verbose=False)
    assert out.wout.volume == pytest.approx(0.307272962641336, rel=5e-5)
    assert out.wout.aspect == pytest.approx(5.43497448141081, rel=5e-6)
    assert out.wout.wb == pytest.approx(0.00128358507023996, rel=5e-6)


def test_free_boundary_wout_fields_match_reference():
    # Converged free-boundary wout against the committed reference,
    # per-field tolerances in the drift family (loosest for the
    # finite-difference current densities). Requires at least eight
    # fields to compare.
    _single_env()
    vi = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")
    vi.mgrid_file = str(TEST_DATA_DIR / "mgrid_cth_like.nc")
    out = vmecpp.run(vi, max_threads=1, verbose=False)
    ref = netCDF4.Dataset(TEST_DATA_DIR / "wout_cth_like_free_bdy.nc")

    fields = [
        ("volume_p", "volume", 5e-5),
        ("aspect", "aspect", 5e-6),
        ("wb", "wb", 5e-6),
        ("b0", "b0", 5e-5),
        ("betatotal", "betatotal", 5e-4),
        ("rbtor", "rbtor", 5e-5),
        ("iotaf", "iotaf", 5e-4),
        ("presf", "presf", 5e-4),
        ("phipf", "phipf", 5e-5),
        ("chipf", "chipf", 5e-4),
        ("jcuru", "jcuru", 5e-3),
        ("jcurv", "jcurv", 5e-3),
        ("q_factor", "q_factor", 5e-4),
        ("specw", "specw", 5e-4),
    ]
    compared = 0
    for nc_name, attr, rel in fields:
        if nc_name not in ref.variables or not hasattr(out.wout, attr):
            continue
        expected = np.asarray(ref[nc_name][...], dtype=float)
        actual = np.asarray(getattr(out.wout, attr), dtype=float)
        assert expected.shape == actual.shape, (nc_name, expected.shape, actual.shape)
        scale = max(float(np.max(np.abs(expected))), 1e-30)
        err = float(np.max(np.abs(actual - expected))) / scale
        assert err <= rel, f"{nc_name}: max rel err {err:.3e} > {rel:.1e}"
        compared += 1
    assert compared >= 8, f"only {compared} fields compared"


def test_free_boundary_broadcast_matches_single():
    # Free-boundary broadcast: per-configuration vacuum solves under a
    # batch-wide activation gate and ivacskip cadence. Configuration
    # zero's result agrees with the single run far inside the
    # free-boundary drift family (the n > 1 residual and controller
    # arithmetic differs from the single path at the ulp level and the
    # free-boundary trajectory carries it).
    _single_env()
    vi = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")
    vi.mgrid_file = str(TEST_DATA_DIR / "mgrid_cth_like.nc")
    single = vmecpp.run(vi, max_threads=1, verbose=False)
    outs = _vmecpp.run_batched_gpu(
        [vi._to_cpp_vmecindata()] * 2, max_threads=1,
        verbose=_vmecpp.OutputMode.SILENT
    )
    assert len(outs) == 1
    assert outs[0].wout.volume == pytest.approx(single.wout.volume, rel=1e-10)
    assert outs[0].wout.wb == pytest.approx(single.wout.wb, rel=1e-10)


def test_free_boundary_distinct_two_boundaries():
    # Distinct free-boundary batch: two boundaries against one coil set
    # converge to their own equilibria in one batched run. The batch
    # shares the vacuum activation iteration and the NESTOR cadence, so
    # each configuration sits in the drift family of its single run.
    _single_env()
    os.environ["VMECPP_BATCH_DISTINCT"] = "1"
    try:
        a = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")
        a.mgrid_file = str(TEST_DATA_DIR / "mgrid_cth_like.nc")
        b = a.model_copy(deep=True)
        b.rbc = b.rbc * 1.005
        b.zbs = b.zbs * 1.005
        outs = _vmecpp.run_batched_gpu(
            [a._to_cpp_vmecindata(), b._to_cpp_vmecindata()], max_threads=1,
            verbose=_vmecpp.OutputMode.SILENT
        )
    finally:
        os.environ.pop("VMECPP_BATCH_DISTINCT", None)
    assert len(outs) == 2
    assert outs[0].wout.volume == pytest.approx(0.307272962641336, rel=5e-4)
    assert outs[1].wout.volume > outs[0].wout.volume


def test_fixed_then_free_boundary_one_process():
    # A fixed-boundary run followed by a free-boundary run in the same
    # process: the vacuum block's host-side consumers (the edge-pressure
    # extrapolation reads presH) must see flushed device values, not
    # whatever the reallocated host arrays inherited from the prior run.
    _single_env()
    vmecpp.run(_cma_input(), max_threads=1, verbose=False)
    vi = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")
    vi.mgrid_file = str(TEST_DATA_DIR / "mgrid_cth_like.nc")
    out = vmecpp.run(vi, max_threads=1, verbose=False)
    assert out.wout.volume == pytest.approx(0.307272962641336, rel=5e-5)
    assert out.wout.wb == pytest.approx(0.00128358507023996, rel=5e-6)


def test_hot_restart_reproduces_converged_state():
    # A run restarted from its own converged output must terminate at
    # the same equilibrium; the restart path stages the device state
    # from the hot-restart host vectors rather than the cold initial
    # guess. Hot restart requires a single-stage ns_array.
    _single_env()
    vi = _cma_input()
    vi.ns_array = vi.ns_array[-1:]
    vi.ftol_array = vi.ftol_array[-1:]
    vi.niter_array = vi.niter_array[-1:]
    cold = vmecpp.run(vi, max_threads=1, verbose=False)
    hot = vmecpp.run(vi, max_threads=1, verbose=False, restart_from=cold)
    assert hot.wout.volume == pytest.approx(cold.wout.volume, rel=1e-11)
    assert hot.wout.aspect == pytest.approx(cold.wout.aspect, rel=1e-11)


@pytest.mark.skipif(
    os.environ.get("VMECPP_TEST_CUDA_SLOW") != "1",
    reason="W7-X takes several minutes of GPU time; opt in via VMECPP_TEST_CUDA_SLOW=1",
)
def test_w7x_bad_jacobian_recovery_converges():
    # W7-X exercises the mid-run bad-Jacobian recovery path. The CUDA
    # build converges to the CPU build's equilibrium; the iteration count
    # differs within the documented drift family, so the assertions pin
    # the converged scalars rather than the trajectory.
    _single_env()
    vi = vmecpp.VmecInput.from_file(W7X_JSON)
    out = vmecpp.run(vi, max_threads=1, verbose=False)
    assert out.wout.wb == pytest.approx(1.858625991022547, rel=1e-7)
    assert out.wout.volume == pytest.approx(27.782541690234815, rel=1e-7)
    assert out.wout.aspect == pytest.approx(10.972491914829975, rel=1e-7)


@pytest.mark.skipif(
    os.environ.get("VMECPP_TEST_CUDA_SLOW") != "1",
    reason="W7-X takes several minutes of GPU time; opt in via VMECPP_TEST_CUDA_SLOW=1",
)
def test_wmma_scatter_w7x_full_spectrum():
    # W7-X carries mpol = 12, the wmma tile capacity; the combine pass
    # must accumulate every poloidal mode or the run stalls above ftol.
    _single_env()
    os.environ["VMECPP_SCATTER_CUSTOM_GEMM_WMMA"] = "1"
    try:
        vi = vmecpp.VmecInput.from_file(W7X_JSON)
        out = vmecpp.run(vi, max_threads=1, verbose=False)
    finally:
        os.environ.pop("VMECPP_SCATTER_CUSTOM_GEMM_WMMA", None)
    assert out.wout.volume == pytest.approx(27.782541690234815, rel=1e-7)
    assert out.wout.wb == pytest.approx(1.858625991022547, rel=1e-7)
