#!/usr/bin/env python3
"""Compile one author's solutions into shared objects and write the override manifest that
e2e/override/ loads at runtime.

The input is bench-trace/solutions/<dataset>/<author>/gemm/*.json -- what the MCP server
persisted (and `bench_fleet.py --sync-solutions` pulled back) for each definition the agent
submitted, or a hand-written / reference-scalar set. Each solution carries its own harness
(gemm.h, gemm.cpp) and kernel.cpp, so it is compiled exactly as the harness compiled it:
-O3 + the ISA's -march. The emitted manifest is

  {"framework":"llamacpp", "isa":..., "march":..., "author":...,
   "kernels":[{"op":"mul_mat","type":"q4_K","K":2560,"N":9216,
               "so":".../gemm_q4_K_n9216_k2560.so",
               "symbol":"armbench_entry_gemm","abi":"entry",
               "embeddable":true, "definition":..., "version":...}],
   "skipped":[{"definition":..., "embeddable":false, "reason":...}, ...]}

Which definitions are built: those tagged `e2e:<model>` in bench-trace/definitions/. `kernel="all"`
builds every one the author submitted, `kernel=[NAME, ...]` just those. N and K are the definition's const
axes and the ggml type is the `layout` of its B input in the workloads; only q4_K / q5_K / q6_K gemms
are spliceable (packed ABI). The rest, and any kernel that calls ggml's own routines, are listed under
"skipped" with a reason.
An entry_rows kernel whose own `static_assert(N % d == 0)` the override granularity does not cover
also gets "row_multiple" (kernel_row_multiple); the hook rounds its row ranges to it.

  build_kernel_set("AUTHOR", "~/e2e/manifest.json", model="qwen3.5-4b", isa="sve2")
  # or: python -m e2e evaluate --model qwen3.5-4b --steps kernel_set --author AUTHOR --rows-abi

Guards, checked after the manifest is built (KernelSet.guard_failures lists what did not hold):
  expect_count=N                the manifest holds exactly N kernels
  reference_author=A2           compare against author A2's solutions for the same definitions
    expect_changed=[NAME, ...]  each of these must differ from A2's
    expect_same=[NAME, ...]     each of these must be byte-identical
  forbid=[REGEX, ...]           no kernel's source may match
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from contracts import _isa_table  # noqa: E402  (config/kernel_contracts.yaml isa table)
from e2e.errors import StepFailed  # noqa: E402
from e2e.common.build import compile_shared, undefined_symbols  # noqa: E402
from e2e.common.guards import check_guards  # noqa: E402
from e2e.common.solutions import SOLUTIONS, load_author, unsubmitted  # noqa: E402

SPLICEABLE_TYPES = ("q4_K", "q5_K", "q6_K")   # the types the override hook has a kernel path for
FRAMEWORK = "llamacpp"


RT_GEMM_CPP = """// Binding harness, runtime-N variant (rows ABI).
// N lives in a thread_local so the e2e override can hand each ggml thread a row range:
//   armbench_entry_gemm(A, out, B, M)             -> all N_FULL rows (out stride N_FULL)
//   armbench_entry_gemm_rows(A, out, B, M, n_rows) -> rows [0, n_rows) of B, out stride n_rows
#include "gemm.h"
namespace gemm_def { thread_local int N = N_FULL; }
using namespace gemm_def;

extern "C" int armbench_entry_gemm(const uint16_t* A, float* output, const uint8_t* B_blocks, int M)
{
    gemm_def::N = N_FULL;
    inner_gemm(A, output, B_blocks, M);
    return 0;
}
extern "C" int armbench_entry_gemm_rows(const uint16_t* A, float* output, const uint8_t* B_blocks, int M, int n_rows)
{
    gemm_def::N = n_rows;
    inner_gemm(A, output, B_blocks, M);
    return 0;
}
"""


# Every row range the override's N-split hands to armbench_entry_gemm_rows is a multiple of this many
# rows (ARMBENCH_N_CHUNK in override/armbench_override.c; 128 when M > 1).
ROWS_GRANULE = 64

# static_assert(N % d == 0 ...), d an integer literal or a name
N_ASSERT_RX = re.compile(r"static_assert\(\s*N\s*%\s*(\d+|[A-Za-z_]\w*)\s*==\s*0")
# constexpr <type> NAME = <integer literal>;   (e.g. `constexpr int NCR = 128;`)
CONSTEXPR_INT_RX = re.compile(r"\bconstexpr\s+(?:[A-Za-z_][\w:]*\s+)+([A-Za-z_]\w*)\s*=\s*(\d+)[uUlL]*\s*;")


def n_divisor(src: str, d: str) -> int | None:
    """The value of divisor `d` in a `static_assert(N % d == 0)` of `src`: a literal, or a name
    with a single `constexpr` integer value in the source. None if it cannot be told for sure
    (no or several definitions, an expression, a macro, 0)."""
    if d.isdigit():
        v = int(d)
    else:
        vals = {int(m.group(2)) for m in CONSTEXPR_INT_RX.finditer(src) if m.group(1) == d}
        if len(vals) != 1:
            return None
        v = vals.pop()
    return v or None


def kernel_row_multiple(src: str) -> int:
    """The rows each N-split range must be a multiple of for kernel.cpp `src` to hold its own
    `static_assert(N % d == 0)`s: the lcm of those d (literal or named, see n_divisor). The
    manifest records it as "row_multiple" when the override's own granularity (ROWS_GRANULE)
    does not already guarantee it, and the override then rounds this kernel's ranges to it."""
    r = 1
    for m in N_ASSERT_RX.finditer(src):
        v = n_divisor(src, m.group(1))
        if v:
            r = math.lcm(r, v)
    return r


def kernel_rows_abi_text(src: str) -> str:
    """The kernel.cpp rewrite the rows ABI needs. The override calls the kernel from every ggml
    thread at once, each with its own row range, so:
      - mutable static state becomes thread_local: scratch vectors, arrays, and scalars or
        pointers such as a lazily grown `static void* buf; static size_t cap;` (const lookup
        tables are left alone);
      - `static_assert(N % d == 0)` checks N_FULL instead of the runtime N, for an integer
        literal d or a name d with one constexpr integer value (otherwise it would not compile
        and the kernel would silently fall back to the single-thread entry). Every row range the
        override hands out is then a multiple of d: of ROWS_GRANULE anyway, or of the kernel's
        manifest "row_multiple" (kernel_row_multiple). A d whose value cannot be told is left
        alone, so that kernel keeps the single-thread entry.
    Idempotent, so it doubles as the normaliser the --expect-* guards compare sources through."""
    src = re.sub(r"\bstatic\s+std::vector", "static thread_local std::vector", src)
    src = re.sub(r"\bstatic\s+(?!const\b|constexpr\b|inline\b|thread_local\b)((?:[A-Za-z_][A-Za-z0-9_:<>]*\s+)+)([A-Za-z_][A-Za-z0-9_]*\s*\[)",
                 r"static thread_local \1\2", src)
    # scalars and pointers: a name followed by `=` or `;` (function declarations have `(` there)
    src = re.sub(r"\bstatic\s+(?!const\b|constexpr\b|inline\b|thread_local\b)((?:[A-Za-z_][\w:<>,]*[\s*&]+)+)([A-Za-z_]\w*\s*(?:=|;))",
                 r"static thread_local \1\2", src)
    src = N_ASSERT_RX.sub(lambda m: m.group(0).replace("N", "N_FULL", 1) if n_divisor(src, m.group(1)) else m.group(0),
                          src)
    return src


def manifest_row_multiple(src: str) -> int:
    """kernel_row_multiple(src) if the override has to round this kernel's row ranges to it,
    else 1 (ROWS_GRANULE already is a multiple of it, the manifest entry gets no key)."""
    r = kernel_row_multiple(src)
    return 1 if ROWS_GRANULE % r == 0 else r


def rt_gemm_cpp(row_multiple: int) -> str:
    """RT_GEMM_CPP; for row_multiple > 1 the rows entry also refuses (returns 1, which the override
    turns into an abort) a row count that is not a multiple of it, so an override that ignores the
    manifest's "row_multiple" fails loudly instead of running the kernel on rows it rejects."""
    if row_multiple == 1:
        return RT_GEMM_CPP
    old = "    gemm_def::N = n_rows;\n"
    assert RT_GEMM_CPP.count(old) == 1
    return RT_GEMM_CPP.replace(old, f"    if (n_rows % {row_multiple} != 0) return 1;  // manifest row_multiple\n" + old)


def rows_abi_sources(build: Path, N: int) -> int:
    """Rewrite the materialized harness so N is a runtime thread_local (see RT_GEMM_CPP)
    and make the kernel's static scratch buffers thread-local. Returns the manifest
    row_multiple (manifest_row_multiple)."""
    h = (build / "gemm.h").read_text()
    old = f"constexpr int N = {N};"
    assert old in h, "gemm.h layout changed; cannot rewrite N"
    h = h.replace(old, f"constexpr int N_FULL = {N};\nextern thread_local int N;   // runtime (rows-abi), == N_FULL unless the override splits rows")
    (build / "gemm.h").write_text(h)
    src = (build / "kernel.cpp").read_text()
    row_multiple = manifest_row_multiple(src)
    (build / "gemm.cpp").write_text(rt_gemm_cpp(row_multiple))
    (build / "kernel.cpp").write_text(kernel_rows_abi_text(src))
    return row_multiple


def e2e_definitions(root: Path, model: str) -> dict:
    """{definition name: {"type", "N", "K"}} for the definitions under the warehouse `root` that carry the
    tag `e2e:<model>`. "type" is the ggml type from the layout of the B input in the definition's
    workloads (None if there is none); N and K are the definition's const axes (None if not const)."""
    found = {}
    for f in sorted((Path(root) / "definitions").glob("*/*.json")):
        d = json.loads(f.read_text())
        if f"e2e:{model}" not in d.get("tags", []):
            continue
        axes = d.get("axes", {})
        wl = Path(root) / "workloads" / d["op_type"] / f"{d['name']}.jsonl"
        first = next((json.loads(line) for line in wl.open() if line.strip()), {}) if wl.exists() else {}
        layout = first.get("inputs", {}).get("B", {}).get("layout", "")
        found[d["name"]] = {"type": layout[len("ggml_"):] if layout.startswith("ggml_") else None,
                            "N": axes.get("N", {}).get("value"), "K": axes.get("K", {}).get("value")}
    return found


def embeddable(info: dict) -> tuple[bool, str]:
    """Can the override hook splice a kernel for this definition into llama.cpp, and if not, why."""
    if info["type"] not in SPLICEABLE_TYPES:
        return False, f"ggml type {info['type']!r} is not one the hook has a kernel for ({' / '.join(SPLICEABLE_TYPES)})"
    if not isinstance(info["N"], int) or not isinstance(info["K"], int):
        return False, "N and K are not both const axes of the definition"
    return True, ""


def digest(text: str) -> str:
    """Hash of a kernel source after the rows-ABI rewrite, so sources compare the same with or without --rows-abi."""
    return hashlib.sha256(kernel_rows_abi_text(text).encode()).hexdigest()[:12]


def compile_so(cxx: str, march: str, build: Path, so: Path):
    return compile_shared(cxx, march, build, [build / "kernel.cpp", build / "gemm.cpp"], so)


def build_kernels(solutions: dict, author: str, out_dir: Path, march: str, *, selected: dict,
                  rows_abi: bool = False, cxx: str = "clang++-18"):
    kernels, skipped = [], []
    for name, sol in sorted(solutions.items()):
        if name not in selected:
            skipped.append({"definition": name, "reason": "not selected (no e2e tag of this model, or not the --kernel asked for)"}); continue
        ok, why = embeddable(selected[name])
        if not ok:
            skipped.append({"definition": name, "embeddable": False, "reason": why}); continue
        qtype, N, K = selected[name]["type"], selected[name]["N"], selected[name]["K"]
        build = out_dir / "build" / name
        build.mkdir(parents=True, exist_ok=True)

        def materialize():
            for s in sol["sources"]:
                (build / s["path"]).write_text(s["content"])
        materialize()
        so = out_dir / f"{name}.so"
        abi, row_multiple = "entry", 1
        if rows_abi:
            row_multiple = rows_abi_sources(build, N)
            p = compile_so(cxx, march, build, so)
            if p.returncode == 0:
                abi = "entry_rows"
            else:
                print(f"  {name}: rows-abi compile failed, falling back to single-thread entry: "
                      f"{p.stderr.strip().splitlines()[-1][:160] if p.stderr.strip() else '?'}")
                materialize()
        if abi == "entry":
            p = compile_so(cxx, march, build, so)
            if p.returncode != 0:
                skipped.append({"definition": name, "embeddable": False, "reason": "compile failed: " + p.stderr[-400:]}); continue
        bad = undefined_symbols(so, "ggml_")
        if bad:
            skipped.append({
                "definition": name, "embeddable": False,
                "reason": ("references ggml internals (" + ", ".join(bad) + "): the kernel "
                           "delegates to ggml's own routine, so it cannot be dlopened outside "
                           "the harness and would disable every override in this manifest"),
            })
            print(f"  {name}: SKIPPED -- calls ggml internals ({', '.join(bad)})")
            continue
        entry = {"op": "mul_mat", "type": qtype, "K": K, "N": N, "so": str(so),
                 "symbol": "armbench_entry_gemm", "abi": abi}
        if abi == "entry_rows" and row_multiple > 1:
            entry["row_multiple"] = row_multiple
        kernels.append({**entry, "embeddable": True, "definition": name, "version": f"solutions/{author}"})
        print(f"built {name} abi={abi}" + (f" row_multiple={row_multiple}" if "row_multiple" in entry else ""))
    return kernels, skipped


@dataclass
class KernelSet:
    manifest: dict
    guard_failures: list


def build_kernel_set(
    author: str, out: Path, *, model: str, isa: str = "sve2", dataset: str = "llama.cpp",
    solutions_root: Path = SOLUTIONS, kernel: str | list[str] = "all",
    cxx: str = "clang++-18", march: str | None = None, rows_abi: bool = False,
    expect_count: int | None = None, reference_author: str | None = None,
    expect_changed=(), expect_same=(), forbid=(),
) -> KernelSet:
    """Compile `author`'s solutions into shared objects and write the override manifest to `out`.

    model            key of config/e2e_models.json; only definitions tagged `e2e:<model>` are considered
    kernel           "all": every such definition that the author submitted; or definition names (one or a
                     list), each of which must carry the tag `e2e:<model>` and have a solution of the author
    march            override the ISA's -march
    rows_abi         recompile each kernel with N as a runtime thread_local and export
                     armbench_entry_gemm_rows(A,out,B,M,n_rows) too (manifest abi 'entry_rows'), so the
                     override can split rows across ggml's threads; a kernel that does not compile that
                     way keeps the single-thread 'entry' ABI
    The remaining keyword arguments are the guards (see the module docstring).
    """
    out = Path(out)
    out_dir = out.parent.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    march = march or _isa_table()[isa].march

    solutions = load_author(dataset, author, solutions_root)
    tagged = e2e_definitions(Path(solutions_root).parent, model)
    names = [kernel] if isinstance(kernel, str) else list(kernel)
    if names == ["all"]:
        selected = tagged
    else:
        untagged = [n for n in names if n not in tagged]
        if untagged:
            raise StepFailed(f"--kernel {untagged} not definitions tagged e2e:{model} under {Path(solutions_root).parent}")
        missing = [n for n in names if n not in solutions]
        if missing:
            raise StepFailed(f"author {author!r} has no solution for {missing} under {solutions_root}/{dataset}/{author}/gemm")
        selected = {n: tagged[n] for n in names}
    kernels, skipped = build_kernels(solutions, author, out_dir, march, selected=selected, rows_abi=rows_abi, cxx=cxx)
    if not kernels:
        raise StepFailed(f"no kernel of {author!r} went into the set for e2e:{model} ({len(skipped)} skipped); "
                         "measure would time an agent build that replaces nothing")
    manifest = {"framework": FRAMEWORK, "isa": isa, "march": march, "author": author,
                "kernels": kernels, "skipped": skipped}
    out.write_text(json.dumps(manifest, indent=1))
    print(f"{len(kernels)} kernels, {len(skipped)} skipped -> {out}")
    for reason in sorted({s["reason"].split(":")[0] for s in skipped}):
        print(f"  skipped: {reason} ({sum(1 for s in skipped if s['reason'].split(':')[0] == reason)})")

    reference = load_author(dataset, reference_author, solutions_root) if reference_author else {}
    failures = check_guards(solutions, [k["definition"] for k in kernels], reference, digest=digest,
                            expect_count=expect_count, expect_changed=expect_changed,
                            expect_same=expect_same, forbid=forbid)
    failures += unsubmitted(dataset, author, solutions_root)
    return KernelSet(manifest, failures)
