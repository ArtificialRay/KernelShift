#!/usr/bin/env python3
"""Rebuild a measure() results JSON from the output measure printed (the
"[  123s] rep 0 stock    t= 1 pp    13.73 tok/s" lines and the "perplexity <build>: ..." lines).
The `report` step falls back to this when the results file is missing and `--log` is given.

  results_from_log("measure_output.log", "e2e_results.json", model="Qwen3.5-4B-Q4_K_M.gguf")
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

SAMPLE = re.compile(r"\[\s*(\d+)s\] rep (\d+) (\S+)\s+t=\s*(\d+) (pp|tg)\s+([\d.]+) tok/s(?:\s+\(n=(\d+) ub=(\w+)\))?")
PERPLEXITY = re.compile(r"perplexity (\S+): (.*)")


def results_from_log(stdout_log: Path, out: Path, *, model: str = "", pp: int | list[int] = 512, tg: int = 128) -> dict:
    """Rebuild a measure() results file from its printed output. Writes `out`, returns the results."""
    stdout_log, out = Path(stdout_log), Path(out)
    pps = [pp] if isinstance(pp, int) else list(pp)
    samples, swept = [], False
    for l in stdout_log.read_text().splitlines():
        m = SAMPLE.match(l)
        if m:
            swept = swept or m.group(7) is not None
            ub = None if m.group(8) in (None, "default") else int(m.group(8))
            samples.append({"build": m.group(3), "threads": int(m.group(4)), "rep": int(m.group(2)), "kind": m.group(5), "ub": ub,
                            "n": int(m.group(7) or pps[0]) if m.group(5) == "pp" else tg, "tok_per_s": float(m.group(6)),
                            "t_elapsed_s": int(m.group(1))})
    builds = list(dict.fromkeys(s["build"] for s in samples))

    def cell(name, s):
        """Same summary keys as measure(): one per build, threads and kind, and per pp x ub pair when swept."""
        if not swept or s["kind"] == "tg":
            return f"{name}/t{s['threads']}/{s['kind']}"
        return f"{name}/t{s['threads']}/pp{s['n']}/ub{s['ub'] or 'default'}"

    summary, base_of = {}, {}
    for b in builds:
        xs = {}
        for s in samples:
            if s["build"] == b:
                xs.setdefault(cell(b, s), []).append(s["tok_per_s"])
                base_of[cell(b, s)] = cell("stock", s)
        for k, v in xs.items():
            summary[k] = {"median": statistics.median(v), "max": max(v), "min": min(v), "n": len(v)}
    for key, v in summary.items():
        base = summary.get(base_of[key])
        if base and not key.startswith("stock/"):
            v["speedup_vs_stock_median"] = v["median"] / base["median"]

    ppl = {m.group(1): m.group(2).strip() for m in map(PERPLEXITY.match, stdout_log.read_text().splitlines()) if m}

    reps = max((s["rep"] for s in samples), default=-1) + 1
    res = {"model": model, "pp": pps[0] if len(pps) == 1 else pps, "tg": tg, "reps": reps, "builds": {b: {} for b in builds}, "host": "",
           "summary": summary, "perplexity": ppl, "samples": samples,
           "note": "rebuilt from the printed measure output"}
    out.write_text(json.dumps(res, indent=1))
    print(f"{len(samples)} samples, {len(summary)} configs, reps={reps}, perplexity for {list(ppl)} -> {out}")
    return res
