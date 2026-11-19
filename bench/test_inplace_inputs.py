"""In-place kernels (loop_120-124) must sort a fresh input on every timed call (#124).

    python3 bench/test_inplace_inputs.py      # needs a C compiler (cc)
"""
import ctypes
import subprocess
import tempfile
import types
from pathlib import Path

import numpy as np

from bench.config import BenchmarkConfig
from bench.data.definition import Definition
from bench.data.trace import Environment
from bench.datasets.simd_loop import SimdLoopDataset
from bench.evaluators.default import DefaultEvaluator
from bench.evaluators.evaluator import BoundKernel

# Insertion sort that returns at once on sorted input, the shortcut that made the old timing meaningless.
KERNEL = r"""
#include <stdint.h>
long saw_sorted = 0, calls = 0;
int armbench_entry_loop_120(void* d, int64_t n, void* res) {
  int32_t* a = (int32_t*)d; int sorted = 1; calls++;
  for (int64_t i = 1; i < n; i++) if (a[i-1] > a[i]) { sorted = 0; break; }
  if (sorted) { saw_sorted++; return 0; }
  for (int64_t i = 1; i < n; i++) { int32_t v = a[i]; int64_t j = i - 1; while (j >= 0 && a[j] > v) { a[j+1] = a[j]; j--; } a[j+1] = v; }
  return 0;
}
"""

DEFINITION = {
    "name": "loop_120", "op_type": "loop_120", "tags": ["simd-loop"],
    "axes": {"N": {"type": "var"}},
    "inputs": {"data": {"shape": ["N"], "dtype": "int32"}},
    "outputs": {"sorted_data": {"shape": ["N"], "dtype": "int32"}},
    "reference": "import numpy as np\n\ndef run(data):\n    return np.sort(data)\n",
    "simd_loop_meta": {"output_inplace": True, "array_pad": 0, "scratch": []},
}


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "k.c").write_text(KERNEL)
    subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", str(tmp / "k.so"), str(tmp / "k.c")], check=True)
    lib = ctypes.CDLL(str(tmp / "k.so"))
    entry = lib.armbench_entry_loop_120
    entry._lib = lib
    count = lambda name: ctypes.c_long.in_dll(lib, name).value  # noqa: E731

    d = Definition.model_validate(DEFINITION)
    x = np.random.default_rng(0).integers(-1000, 1000, 2000).astype(np.int32)
    base = types.SimpleNamespace(np_inputs={"data": x}, ref_np=np.sort(x))
    cfg = BenchmarkConfig(baseline_author="baseline-sve").resolve_eval_config(d)
    env = Environment(hardware="test")
    kernel = BoundKernel(entry=entry, adapter=SimdLoopDataset(), op_type="loop_120")

    ctx = kernel.prepare(base.np_inputs, d, out_shape=base.ref_np.shape)
    _, ev = DefaultEvaluator.check_correctness(d, kernel, ctx, base, cfg, env, "t")
    assert ev is None, ev.log
    perf, ev = DefaultEvaluator.eval_performance(d, None, kernel, ctx, base, cfg, env, "t",
                                                 is_baseline=True, trace_set=None)
    assert ev is None, ev.log
    assert count("saw_sorted") == 0, f"{count('saw_sorted')} of {count('calls')} calls got sorted input"
    assert perf.inner_iters == 1

    # A kernel that is right only on its first call fails the check after timing.
    noop = BoundKernel(entry=entry, adapter=SimdLoopDataset(), op_type="loop_120")
    noop.invoke = lambda c: 0
    _, ev = DefaultEvaluator.eval_performance(d, None, noop, ctx, base, cfg, env, "t",
                                              is_baseline=True, trace_set=None)
    assert ev is not None and ev.status.value == "INCORRECT_NUMERICAL"
    print(f"ok: {count('calls')} calls, none on sorted input")


if __name__ == "__main__":
    main()
