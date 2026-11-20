#!/usr/bin/env python3
"""Render a measure.py results JSON as one results table.

    render_report("e2e_results.json", baseline="stock")      # or: python -m e2e evaluate --steps report
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path


def pp_config(s):
    """(prompt length, micro-batch) a prompt-processing sample was timed with; None for generation."""
    return (s["n"], s.get("ub")) if s["kind"] == "pp" else None


def medians(samples):
    """(build, threads, kind, pp_config) -> median tok/s and number of samples."""
    out = {}
    for s in samples:
        out.setdefault((s["build"], s["threads"], s["kind"], pp_config(s)), []).append(s["tok_per_s"])
    return {k: statistics.median(v) for k, v in out.items()}, {k: len(v) for k, v in out.items()}


def render_report(results, baseline: str = "stock") -> str:
    """The table of a measure() results file (a path, or the dict measure() returned):
    throughput, speedup vs `baseline`, and perplexity."""
    lines: list[str] = []
    say = lines.append
    d = results if isinstance(results, dict) else json.loads(Path(results).read_text())
    med, n = medians(d.get("samples", []))
    builds = list(dict.fromkeys(s["build"] for s in d.get("samples", [])))
    threads = sorted({s["threads"] for s in d.get("samples", [])})
    configs = list(dict.fromkeys(pp_config(s) for s in d.get("samples", []) if s["kind"] == "pp"))
    main = configs[0] if configs else None   # tg was timed with the first pair
    ppl = d.get("perplexity", {}) or {}

    say(f"host: {d.get('host','?')}")
    say(f"model: {Path(d.get('model','?')).name}   pp={d.get('pp')} tg={d.get('tg')} ub={d.get('ub')} reps={d.get('reps')}\n")

    say("absolute throughput (tok/s, median of reps)" + (f"; pp {main[0]}, ub {main[1] or 'default'}" if len(configs) > 1 else ""))
    hdr = "build".ljust(18) + "".join(f"  t={t:<3} pp      tg" for t in threads)
    say(hdr); say("-" * len(hdr))
    for b in builds:
        row = b.ljust(18)
        for t in threads:
            pp = med.get((b, t, "pp", main)); tg = med.get((b, t, "tg", None))
            row += f"  {pp:7.1f} {tg:7.2f}" if pp and tg else "        -       -"
        say(row)

    if baseline in builds:
        say(f"\nspeedup vs {baseline}")
        say(hdr); say("-" * len(hdr))
        for b in builds:
            if b == baseline:
                continue
            row = b.ljust(18)
            for t in threads:
                num_pp, den_pp = med.get((b, t, "pp", main)), med.get((baseline, t, "pp", main))
                num_tg, den_tg = med.get((b, t, "tg", None)), med.get((baseline, t, "tg", None))
                row += (f"  {num_pp/den_pp:6.2f}x {num_tg/den_tg:6.2f}x"
                        if num_pp and den_pp and num_tg and den_tg else "        -       -")
            say(row)

    if len(configs) > 1:
        say("\nprompt processing sweep (tok/s; speedup vs baseline in brackets)")
        for t in threads:
            hdr = f"t={t}".ljust(18) + "".join(f"  pp{n}/ub{u or 'default'}".rjust(18) for n, u in configs)
            say(hdr); say("-" * len(hdr))
            for b in builds:
                row = b.ljust(18)
                for c in configs:
                    v, base = med.get((b, t, "pp", c)), med.get((baseline, t, "pp", c))
                    cell = "-" if v is None else f"{v:.1f}" + (f" [{v/base:.2f}x]" if base and b != baseline else "")
                    row += cell.rjust(18)
                say(row)

    def ppl_of(name):
        """measure_e2e stores llama-perplexity's raw tail, so pull the number out of it."""
        v = ppl.get(name)
        if isinstance(v, dict):
            v = v.get("ppl", v.get("value"))
        if v is None:
            return None
        m = re.search(r"PPL\s*=\s*([0-9]+\.?[0-9]*)", str(v))
        if m:
            return float(m.group(1))
        try:
            return float(str(v).split()[0])
        except (TypeError, ValueError, IndexError):
            return None

    if ppl:
        say("\nperplexity (wikitext-2)")
        base_v = ppl_of(baseline)
        for b in builds:
            v = ppl_of(b)
            if v is None:
                continue
            delta = f"  ({(v / base_v - 1) * 100:+.1f}% vs {baseline})" if base_v else ""
            say(f"  {b.ljust(18)} {v:8.3f}{delta}")

    kld = d.get("kld") or {}
    if kld:
        say(f"\nagreement with {baseline} on {d.get('kld_chunks')} chunks (KL divergence: 0 = identical output)")
        cols = f"{'build':18} {'mean KLD':>10} {'99% KLD':>10} {'RMS dp %':>9} {'same top-1 %':>13} {'PPL ratio':>10}"
        say(cols); say("-" * len(cols))
        for b in builds:
            v = kld.get(b)
            if v is None or b == baseline:
                continue
            say(f"{b:18} {v['mean_kld']:10.6f} {v['p99_kld']:10.6f} {v['rms_dp_pct']:9.3f} {v['same_top_p_pct']:13.3f} {v['ppl_ratio']:10.4f}"
                if "error" not in v else f"{b:18} failed: {v['error'][:80]}")

    short = [k for k, c in n.items() if c < (d.get("reps") or 0)]
    if short:
        say(f"\nwarning: {len(short)} (build, threads, kind) cells have fewer than {d.get('reps')} samples")
    return "\n".join(lines)


def render_kvr(path) -> str:
    """The table of the kvr step's kvr.jsonl: one line per kernel and M, single-threaded.
    repack / norepack / agent are the best time in us of ggml's repacked path, ggml without repack and the
    agent's kernel; a/repack and a/norepack are speedups of the agent over each (> 1: the agent is faster);
    sqnr is the agent's output against the repack path, in dB."""
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    f = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"
    hdr = (f"{'kernel':34} {'M':>4} {'repack':>9} {'norepack':>9} {'agent':>9} {'a/repack':>9} "
           f"{'a/norepack':>10} {'sqnr':>6}")
    lines = ["per kernel, single thread (us; speedup of the agent over ggml's repack / norepack path)", hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(f"{r['shape']:34} {r['M']:>4} {f(r['t_repack_min_us']):>9} {f(r['t_norepack_min_us']):>9} "
                     f"{f(r['t_agent_min_us']):>9} {f(r['agent_vs_repack'], 2):>9} {f(r['agent_vs_norepack'], 2):>10} "
                     f"{f(r.get('sqnr_db', r.get('sqnr_agent_vs_repack_db'))):>6}" + ("" if r["repack_used"] else "   (no repack kernel for this shape)"))
    return "\n".join(lines)
