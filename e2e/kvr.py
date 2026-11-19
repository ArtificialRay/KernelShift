#!/usr/bin/env python3
"""Per-kernel timing of the agent's kernels against llama.cpp's own paths (the `kvr` step).

For every mul_mat kernel of the run's manifest.json and every M, one single-threaded timing of the same
weights and activations through (a) ggml's repacked CPU path (what stock llama.cpp runs), (b) ggml without
repack (what the norepack build runs) and (c) the agent's .so; plus how far the agent's output is from
them (SQNR). The harness is e2e/kvr/ (bench_repack.cpp, see e2e/kvr/README.md); this module builds it
against the box's build-stock and runs it for the whole manifest.

Run on the box: `python3 -m e2e evaluate --steps kvr ...`.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from e2e.errors import StepFailed

KVR_DIR = Path(__file__).resolve().parent / "kvr"
DEFAULT_MS = (1, 2, 4, 8, 16, 32, 128, 512)


def kvr(manifest: Path, llama_dir: Path, out: Path, *, ms=DEFAULT_MS, work_dir: Path | None = None) -> list[dict]:
    """Build the harness against `llama_dir`/build-stock, time every mul_mat kernel of `manifest`
    for each M in `ms`, write one JSON line per (shape, M) to `out` and return them."""
    manifest, llama_dir, out = Path(manifest), Path(llama_dir).expanduser(), Path(out)
    build_stock = llama_dir / "build-stock"
    if not build_stock.is_dir():
        raise StepFailed(f"no build-stock under {llama_dir}; run the deploy step first")
    if not manifest.exists():
        raise StepFailed(f"{manifest} does not exist; run the kernel_set step first")
    work = Path(work_dir or out.parent / "build")
    work.mkdir(parents=True, exist_ok=True)
    binary = work / "bench_repack"

    # the binary is linked against build-stock, which has repack on (the others have it off, so no repack buffer)
    subprocess.run(["bash", str(KVR_DIR / "build.sh"), str(llama_dir), str(build_stock), str(binary)], check=False,
                   stdout=(work / "build.log").open("w"), stderr=subprocess.STDOUT)
    if not binary.exists():
        raise StepFailed(f"building the harness failed, see {work / 'build.log'}")

    out.unlink(missing_ok=True)
    env = {**os.environ, "BENCH": str(binary), "MS": ",".join(map(str, ms))}
    env.pop("ARMBENCH_OVERRIDES", None)   # the hook must not claim the plain-buffer baseline
    p = subprocess.run(["bash", str(KVR_DIR / "run_all.sh"), str(manifest), str(out)], env=env, text=True, timeout=7200)
    if p.returncode != 0 or not out.exists():
        raise StepFailed(f"run_all.sh failed (rc={p.returncode})")
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    failed = sorted({r["shape"] for r in rows if "error" in r})
    if not rows or failed:
        raise StepFailed(f"kvr: no result for {failed or 'any shape'} (see {out})")
    return rows
