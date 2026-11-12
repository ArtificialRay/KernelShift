#!/usr/bin/env python3
"""Tests for the rows-ABI rewrite in kernel_set.py (static_assert(N % d == 0) handling).

Run:  python3 e2e/adapters/llamacpp/test_kernel_set.py      (or pytest; the build test needs a C++ compiler,
      $CXX or clang++-18 / clang++ / c++, and is skipped without one)

Checks:
  A. a literal or named (constexpr) divisor of the hook's 64-row granularity is checked against N_FULL and
     adds nothing to the manifest (same as before named divisors were handled);
  B. a named divisor the granularity does not cover (`constexpr int NCR = 128;`) is checked against N_FULL
     and becomes the manifest "row_multiple"; the rows entry refuses other row counts;
  C. a divisor whose value cannot be told (undefined, two values, an expression) is left alone;
  D. the rewrite is idempotent;
  E. build_kernels end to end: the NCR kernel builds as entry_rows with row_multiple 128, its rows entry
     returns 0 for 128 rows and 1 for 64; a kernel without such an assert gets no row_multiple key.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from e2e.adapters.llamacpp import kernel_set as ks  # noqa: E402


def kernel(prelude: str) -> str:
    return prelude + "\nvoid f() { for (int n = 0; n < N; ++n) {} }\n"


def test_divisor_of_granule():
    for prelude in ("static_assert(N % 4 == 0, \"x\");",
                    "constexpr int NR = 8;\nstatic_assert(N % NR == 0, \"x\");"):
        out = ks.kernel_rows_abi_text(kernel(prelude))
        assert "static_assert(N_FULL %" in out, out
        assert ks.manifest_row_multiple(kernel(prelude)) == 1
    assert ks.rt_gemm_cpp(1) == ks.RT_GEMM_CPP


def test_named_divisor_beyond_granule():
    src = kernel("constexpr int NCR = 128;           // rows per chunk\n"
                 "static_assert(N % NCR == 0, \"N must be a multiple of NCR\");")
    out = ks.kernel_rows_abi_text(src)
    assert "static_assert(N_FULL % NCR == 0" in out, out
    assert ks.kernel_row_multiple(src) == 128
    assert ks.manifest_row_multiple(src) == 128
    # two asserts: lcm; a literal beyond the granularity is handled the same way
    two = kernel("constexpr std::size_t A = 96u;\nstatic_assert(N % A == 0, \"\");\nstatic_assert(N % 128 == 0, \"\");")
    assert ks.manifest_row_multiple(two) == 384
    assert ks.kernel_rows_abi_text(two).count("static_assert(N_FULL %") == 2
    g = ks.rt_gemm_cpp(128)
    assert "if (n_rows % 128 != 0) return 1;" in g
    assert g.index("n_rows % 128") > g.index('extern "C" int armbench_entry_gemm_rows')  # rows entry only


def test_unknown_divisor_left_alone():
    for prelude in ("static_assert(N % NCR == 0, \"\");",                                   # no definition
                    "constexpr int NCR = 128;\nnamespace b { constexpr int NCR = 64; }\nstatic_assert(N % NCR == 0, \"\");",
                    "constexpr int NR = 4;\nconstexpr int NCR = 32 * NR;\nstatic_assert(N % NCR == 0, \"\");",
                    "#define NCR 128\nstatic_assert(N % NCR == 0, \"\");",
                    "constexpr int Z = 0;\nstatic_assert(N % Z == 0, \"\");"):
        src = kernel(prelude)
        assert "static_assert(N %" in ks.kernel_rows_abi_text(src), prelude
        assert ks.manifest_row_multiple(src) == 1, prelude


def test_idempotent():
    src = kernel("static std::vector<int> buf;\nstatic float* p = nullptr;\nconstexpr int NCR = 128;\n"
                 "static_assert(N % NCR == 0, \"\");\nstatic_assert(N % 4 == 0, \"\");")
    once = ks.kernel_rows_abi_text(src)
    assert ks.kernel_rows_abi_text(once) == once
    assert ks.digest(src) == ks.digest(once)


GEMM_H = """#pragma once
#include <cstdint>
namespace gemm_def {
constexpr int N = 256;
constexpr int K = 256;
constexpr int K_blk = 1;
constexpr int K_bytes = 144;
} // namespace gemm_def
extern "C" void inner_gemm(const uint16_t* A, float* output, const uint8_t* B_blocks, int M);
"""
GEMM_CPP = """#include "gemm.h"
using namespace gemm_def;
extern "C" int armbench_entry_gemm(const uint16_t* A, float* output, const uint8_t* B_blocks, int M)
{ inner_gemm(A, output, B_blocks, M); return 0; }
"""
KERNEL_CPP = """#include "gemm.h"
using namespace gemm_def;
%s
extern "C" void inner_gemm(const uint16_t* A, float* output, const uint8_t* B_blocks, int M)
{
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) output[(long)m * N + n] = (float)B_blocks[(long)n * K_bytes];
}
"""


def compiler() -> str | None:
    for c in (os.environ.get("CXX"), "clang++-18", "clang++", "c++"):
        if c and shutil.which(c):
            return c
    return None


def test_build_kernels():
    cxx = compiler()
    if not cxx:
        print("SKIP test_build_kernels: no C++ compiler")
        return
    sols = {}
    for name, prelude in (("gemm_ggml_q4_K_n256_k256", "constexpr int NCR = 128;\nstatic_assert(N % NCR == 0, \"\");"),
                          ("gemm_ggml_q6_K_n256_k256", "static_assert(N % 4 == 0, \"\");")):
        sols[name] = {"sources": [{"path": "gemm.h", "content": GEMM_H}, {"path": "gemm.cpp", "content": GEMM_CPP},
                                  {"path": "kernel.cpp", "content": KERNEL_CPP % prelude}]}
    with tempfile.TemporaryDirectory() as d:
        # "-w" stands in for the ISA's -march: the test kernel is plain C++
        kernels, skipped = ks.build_kernels(sols, "test", Path(d), "-w", rows_abi=True, cxx=cxx)
        assert not skipped, skipped
        by = {k["definition"]: k for k in kernels}
        q4, q6 = by["gemm_ggml_q4_K_n256_k256"], by["gemm_ggml_q6_K_n256_k256"]
        assert q4["abi"] == "entry_rows" and q4["row_multiple"] == 128, q4
        assert q6["abi"] == "entry_rows" and "row_multiple" not in q6, q6
        lib = ctypes.CDLL(q4["so"])
        rows = lib.armbench_entry_gemm_rows
        rows.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        B = (ctypes.c_uint8 * (256 * 144))(*[(n % 251) if i == 0 else 0 for n in range(256) for i in range(144)])
        A = (ctypes.c_uint16 * 256)()
        out = (ctypes.c_float * 256)()
        assert rows(A, out, ctypes.addressof(B) + 128 * 144, 1, 128) == 0
        assert [out[i] for i in range(128)] == [float((128 + i) % 251) for i in range(128)]
        assert rows(A, out, B, 1, 64) == 1
        assert lib.armbench_entry_gemm(A, out, B, 1) == 0 and out[255] == float(255 % 251)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
