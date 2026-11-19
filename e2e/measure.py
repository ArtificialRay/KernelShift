#!/usr/bin/env python3
"""End-to-end tokens/s measurement: llama-bench (prompt processing pp and text generation tg)
for one or more llama.cpp builds on the same box and model, repeated and interleaved across builds.
Writes one JSON with every raw sample plus median/min summaries, and optionally runs
llama-perplexity on a text file for each build.

Run it on the Graviton box (`python3 -m e2e evaluate --steps measure ...`); it calls measure() with
the stock / norepack / agent builds under ~/llama.cpp-e2e and the run directory's manifest.json as
the agent's override (see e2e/override/README.md).

Before any timing, every build that has an override is run once with ARMBENCH_OVERRIDE_LOG=1 and
must hit every mul_mat shape in its manifest. Override runs set ARMBENCH_OVERRIDE_STRICT=1.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from e2e.errors import StepFailed


def run(cmd, env=None, timeout=3600):
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-4000:])
        raise StepFailed(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}")
    return p.stdout


def llama_bench(bench_bin: Path, model: Path, threads: int, pp: int, tg: int, env: dict,
                fa: str, ub: int | None) -> list[dict]:
    """One llama-bench invocation → list of result dicts (llama-bench -o json)."""
    cmd = [str(bench_bin), "-m", str(model), "-t", str(threads), "-p", str(pp), "-n", str(tg),
           "-fa", fa, "-r", "1", "-o", "json"]
    if ub:
        cmd += ["-ub", str(ub)]
    out = run(cmd, env=env)
    start = out.find("[")
    return json.loads(out[start:])


def llama_commit(build_dir: Path) -> str | None:
    """Short commit of the llama.cpp checkout the build dir sits in (None if it is not a git tree)."""
    p = subprocess.run(["git", "-C", str(build_dir.parent), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() or None


HIT_RX = re.compile(r"hit mul_mat type=(\S+) K=(\d+) N=(\d+)")


def verify_hook(name: str, b: dict, model: Path, threads: int, fa: str, ub: int | None, log: Path) -> dict:
    """Short pp+tg run with logging on; every mul_mat entry of the manifest must be hit.
    The hook's own log lines (which kernels, which ABI, how each call was split) are kept in `log`."""
    want = {(k["type"], int(k["K"]), int(k["N"]))
            for k in json.loads(Path(b["manifest"]).read_text())["kernels"] if k.get("op") == "mul_mat"}
    if not want:
        raise StepFailed(f"{name}: {b['manifest']} has no mul_mat kernel, the override would replace nothing")
    env = {**os.environ, **b["env"], "ARMBENCH_OVERRIDES": b["manifest"], "ARMBENCH_OVERRIDE_LOG": "1",
           "ARMBENCH_OVERRIDE_STRICT": "1"}
    cmd = [str(b["dir"] / "bin" / "llama-bench"), "-m", str(model), "-t", str(threads),
           "-p", "64", "-n", "4", "-fa", fa, "-r", "1"]
    if ub:
        cmd += ["-ub", str(ub)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    log.write_text("".join(l + "\n" for l in p.stderr.splitlines() if l.startswith("[armbench-override]")) + p.stdout)
    hit = {(t, int(k), int(n)) for t, k, n in HIT_RX.findall(p.stderr)}
    missing = sorted(want - hit)
    if p.returncode != 0 or missing:
        sys.stderr.write(p.stderr[-4000:])
        raise StepFailed(f"{name}: override hook check failed (rc={p.returncode}, {len(hit & want)}/{len(want)} "
                         f"manifest shapes hit; missing {missing})")
    print(f"hook {name}: {len(want)}/{len(want)} manifest shapes intercepted", flush=True)
    return {"expected": len(want), "hit": len(hit & want)}


KLD_FIELDS = {   # llama-perplexity --kl-divergence summary line -> key in the results
    "mean_kld": r"Mean\s+KLD:\s*(-?[\d.]+)",
    "median_kld": r"Median\s+KLD:\s*(-?[\d.]+)",
    "p99_kld": r"99\.0%\s+KLD:\s*(-?[\d.]+)",
    "max_kld": r"Maximum KLD:\s*(-?[\d.]+)",
    "rms_dp_pct": r"RMS Δp\s*:\s*(-?[\d.]+)",
    "same_top_p_pct": r"Same top p:\s*(-?[\d.]+)",
    "ppl_ratio": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*(-?[\d.]+)",
}


def parse_kld(text: str) -> dict:
    """The numbers of llama-perplexity's KL-divergence summary; raises if one is missing."""
    out = {}
    for key, rx in KLD_FIELDS.items():
        m = re.search(rx, text)
        if not m:
            raise StepFailed(f"no {key!r} in the KL-divergence output")
        out[key] = float(m.group(1))
    return out


def run_kld(b: dict, model: Path, text: Path, chunks: int, threads: int, fa: str, ub: int | None,
            base: Path, log: Path, reference: bool) -> dict:
    """llama-perplexity on `text` with one build. The reference build saves its logits to `base`
    ({}); every other build is compared with them (KL divergence, top-token agreement)."""
    env = dict(os.environ)
    if b["manifest"]:
        env["ARMBENCH_OVERRIDES"] = b["manifest"]
    env.update(b["env"])
    cmd = [str(b["dir"] / "bin" / "llama-perplexity"), "-m", str(model), "-f", str(text), "--chunks", str(chunks),
           "-t", str(threads), "-fa", fa, "--kl-divergence-base", str(base)]
    if not reference:
        cmd.append("--kl-divergence")
    if ub:
        cmd += ["-ub", str(ub)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=7200)
    log.write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        raise StepFailed(f"rc={p.returncode}: {(p.stdout + p.stderr)[-300:].strip()}")
    return {} if reference else parse_kld(p.stdout + p.stderr)


def measure(
    model: Path, builds: dict, *, overrides: dict | None = None, env: dict | None = None,
    threads=(4,), pp: int | Sequence[int] = 512, tg: int = 128, reps: int = 5, fa: str,
    ub: int | None | Sequence[int] = None,
    text: Path | None = None, ppl_chunks: int = 64, kld_chunks: int | None = None,
    out: Path = Path("e2e_results.json"),
) -> dict:
    """llama-bench (pp / tg) and optionally llama-perplexity for every build, interleaved, on this
    machine; writes the raw samples and summary to `out` and returns them.

    builds      name -> cmake build dir (must contain bin/llama-bench)
    overrides   name -> manifest.json for the builds that run the ARMBENCH_OVERRIDES hook
    env         name -> {KEY: VALUE} extra environment for one build (e.g. the single-thread ablation
                {"agent1t": {"ARMBENCH_OVERRIDE_THREADS": "1"}})
    fa          "on" / "off", passed to llama-bench and llama-perplexity (llama-bench's own default is auto)
    pp          prompt length, or several: prompt processing is timed for every pp x ub pair
    ub          micro-batch size (-ub), or several; llama.cpp's default (512) if None.
                Generation (tg) does not depend on either, so it is timed once, with the first pair.
    text        text file for the quality checks: llama-perplexity on ppl_chunks chunks of it (run with the first ub)
    kld_chunks  if set, also compare every build's output distribution with stock's on the first kld_chunks chunks
                of `text` (KL divergence; stock's saved logits are deleted afterwards)
    """
    model, out, threads = Path(model), Path(out), list(threads)
    pps = [pp] if isinstance(pp, int) else list(pp)
    ubs = [ub] if ub is None or isinstance(ub, int) else (list(ub) or [None])
    configs = [(u, p) for u in ubs for p in pps]   # (ub, pp); the first one also times tg
    swept = len(configs) > 1
    ub = ubs[0]
    builds = {name: {"dir": Path(path).expanduser(), "manifest": None, "env": {}} for name, path in builds.items()}
    for name, path in (overrides or {}).items():
        if name not in builds:
            raise StepFailed(f"overrides names unknown build {name!r}")
        builds[name]["manifest"] = str(Path(path).expanduser())
    for name, extra in (env or {}).items():
        if name not in builds:
            raise StepFailed(f"env names unknown build {name!r}")
        builds[name]["env"].update(extra)
    for name, b in builds.items():
        if not (b["dir"] / "bin" / "llama-bench").exists():
            raise StepFailed(f"{name}: {b['dir']}/bin/llama-bench not found")
    if kld_chunks and not text:
        raise StepFailed("the KL divergence needs a text file: pass --text")
    if kld_chunks and "stock" not in builds:
        raise StepFailed("the KL divergence is measured against the stock build, which is missing")

    for b in builds.values():
        if b["manifest"]:
            b["env"].setdefault("ARMBENCH_OVERRIDE_STRICT", "1")
    hook = {name: verify_hook(name, b, model, max(threads), fa, ub, out.with_name(f"hook_{name}.log"))
            for name, b in builds.items() if b["manifest"]}

    samples = []
    t0 = time.time()
    for rep in range(reps):
        for nthreads in threads:
            for name, b in builds.items():   # interleaved across builds
                env = dict(os.environ)
                if b["manifest"]:
                    env["ARMBENCH_OVERRIDES"] = b["manifest"]
                    env.setdefault("ARMBENCH_OVERRIDE_LOG", "0")
                env.update(b["env"])
                for u, p in configs:
                    for r in llama_bench(b["dir"] / "bin" / "llama-bench", model, nthreads, p,
                                         tg if (u, p) == configs[0] else 0, env, fa, u):
                        kind = "pp" if r.get("n_prompt", 0) > 0 else "tg"
                        samples.append({"build": name, "threads": nthreads, "rep": rep, "kind": kind, "ub": u,
                                        "n": r.get("n_prompt") or r.get("n_gen"),
                                        "tok_per_s": r["avg_ts"], "raw": r})
                        cfg = f"  (n={samples[-1]['n']} ub={u or 'default'})" if swept and kind == "pp" else ""
                        print(f"[{time.time()-t0:6.0f}s] rep {rep} {name:8s} t={nthreads:2d} {kind} {r['avg_ts']:8.2f} tok/s{cfg}", flush=True)

    def cell(name, s):
        """Summary key of a sample: one per build, threads and kind; with several pp x ub pairs, also per pair."""
        if not swept or s["kind"] == "tg":
            return f"{name}/t{s['threads']}/{s['kind']}"
        return f"{name}/t{s['threads']}/pp{s['n']}/ub{s['ub'] or 'default'}"

    summary, speed_base = {}, {}
    for name in builds:
        xs = {}
        for s in samples:
            if s["build"] == name:
                xs.setdefault(cell(name, s), []).append(s["tok_per_s"])
                speed_base[cell(name, s)] = cell("stock", s)
        for k, v in xs.items():
            summary[k] = {"median": statistics.median(v), "max": max(v), "min": min(v), "n": len(v)}
    if "stock" in builds:
        for k, v in summary.items():
            base = summary.get(speed_base[k])
            if base and not k.startswith("stock/"):
                v["speedup_vs_stock_median"] = v["median"] / base["median"]

    kld = {}

    def dump(ppl):
        result = {"model": str(model), "pp": pps[0] if len(pps) == 1 else pps, "tg": tg, "reps": reps,
                  "fa": fa, "ub": ubs[0] if len(ubs) == 1 else ubs, "ppl_chunks": ppl_chunks if text else None,
                  "kld_chunks": kld_chunks, "kld": kld, "hook": hook,
                  "builds": {n: {"dir": str(b["dir"]), "manifest": b["manifest"], "env": b["env"]} for n, b in builds.items()},
                  "host": os.uname().nodename, "date": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
                  "llama_cpp_commit": llama_commit(next(iter(builds.values()))["dir"]), "summary": summary, "perplexity": ppl, "samples": samples}
        out.write_text(json.dumps(result, indent=1))
        return result

    result = dump({})
    print(f"wrote {out} (bench only)", flush=True)

    ppl = {}
    if text:
        for name, b in builds.items():
            env = dict(os.environ)
            if b["manifest"]:
                env["ARMBENCH_OVERRIDES"] = b["manifest"]
            env.update(b["env"])
            try:
                cmd = [str(b["dir"] / "bin" / "llama-perplexity"), "-m", str(model), "-f", str(text),
                       "--chunks", str(ppl_chunks), "-t", str(max(threads)), "-fa", fa]
                if ub:
                    cmd += ["-ub", str(ub)]
                p = subprocess.run(cmd,
                                   capture_output=True, text=True, env=env, timeout=7200)
                output = p.stdout + p.stderr
                line = [l for l in output.splitlines() if "estimate" in l.lower() or "PPL" in l or "Final" in l]
                ppl[name] = (line[-1] if line else output[-300:]).strip() if p.returncode == 0 else f"FAILED rc={p.returncode}: {output[-300:].strip()}"
            except Exception as e:
                ppl[name] = f"FAILED: {e}"
            print(f"perplexity {name}: {ppl[name]}", flush=True)
        result = dump(ppl)

    if kld_chunks:
        base = out.with_name("kld.base")   # stock's logits: gigabytes, so deleted below and never fetched
        try:
            for name in ["stock"] + [n for n in builds if n != "stock"]:
                try:
                    kld[name] = run_kld(builds[name], model, text, kld_chunks, max(threads), fa, ub, base,
                                        out.with_name(f"kld_{name}.log"), reference=name == "stock")
                except Exception as e:
                    kld[name] = {"error": str(e)}
                    if name == "stock":   # nothing to compare the others with
                        break
                print(f"kld {name}: {json.dumps(kld[name])}", flush=True)
        finally:
            base.unlink(missing_ok=True)
        result = dump(ppl)

    print(f"\n{'config':28} {'median tok/s':>13} {'vs stock':>9}")
    for k, v in summary.items():
        print(f"{k:28} {v['median']:13.2f} {v.get('speedup_vs_stock_median', float('nan')):9.3f}")
    print(f"wrote {out}", flush=True)
    return result
