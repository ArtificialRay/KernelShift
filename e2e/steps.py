"""The steps of `python -m e2e evaluate`, and the loops that run them.

Each step is a function from the run context and the options to the arguments of the module
that does the work (kernel_set, deploy, measure, report). Where they run:

  * `deploy` drives the box over SSH, so it runs on the machine the command is typed on;
  * `kernel_set`, `measure` and `report` run on the box. Typed anywhere else, `evaluate` syncs the repo
    to the box, runs them there (e2e.remote) and fetches the run directory; `--here` runs them in
    this process, which is how they run on the box.

A step whose artifact already exists is skipped (ctx.force redoes it); the first failing step stops the run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .context import E2EContext, Options
from .errors import StepFailed


def require(value, option: str, step: str):
    if value is None:
        raise StepFailed(f"the {step} step needs {option}")
    return value


def box_label(ctx: E2EContext, opts: Options) -> str:
    return opts.label or f"e2e-{ctx.model}-{ctx.isa}"


def run_kernel_set(ctx: E2EContext, opts: Options):
    """Compile an author's submitted solutions into the override manifest, then check the guards."""
    from .kernel_set import build_kernel_set
    result = build_kernel_set(
        require(opts.author, "--author", "kernel_set"), ctx.run.manifest, model=ctx.model, isa=ctx.isa,
        solutions_root=opts.root / "solutions", kernel=opts.kernel, cxx=opts.cxx, march=opts.march, rows_abi=opts.rows_abi,
        expect_count=opts.expect_count, reference_author=opts.reference_author,
        expect_changed=opts.expect_changed, expect_same=opts.expect_same, forbid=opts.forbid)
    if result.guard_failures:
        raise StepFailed("\n  " + "\n  ".join(result.guard_failures))
    return result


def run_deploy(ctx: E2EContext, opts: Options):
    """llama.cpp tools tree, the model and (with agent_build) the override build, on the box. Returns the box."""
    from .deploy import deploy
    instance_type = (opts.instance_type or ctx.model_config["min_measure_instance"]) if opts.provision else None
    return deploy(box_label(ctx, opts), ctx.model, instance_type, opts.agent_build)


def run_measure(ctx: E2EContext, opts: Options) -> dict:
    """llama-bench (pp / tg) and llama-perplexity for stock / norepack / agent."""
    from .measure import measure
    llama = Path(opts.llama_dir).expanduser()
    builds = {name: llama / f"build-{name}" for name in ("stock", "norepack", "agent")
              if (llama / f"build-{name}").is_dir()}
    if "stock" not in builds:
        raise StepFailed(f"no build-stock under {llama}; run the deploy step first")
    if "agent" in builds and not ctx.run.manifest.exists():
        raise StepFailed(f"{llama}/build-agent exists but {ctx.run.manifest} does not: it would be timed "
                         "without the override and still be called agent; run the kernel_set step first")
    overrides = {"agent": ctx.run.manifest} if "agent" in builds else {}
    model = Path("~/models").expanduser() / ctx.model_config["file"]
    return measure(model, builds, overrides=overrides, threads=opts.threads, pp=opts.pp, tg=opts.tg,
                   reps=opts.reps, fa=opts.fa, ub=opts.ub,
                   text=Path(opts.text) if opts.text else None,
                   ppl_chunks=opts.ppl_chunks, kld_chunks=opts.kld_chunks if opts.kld else None, out=ctx.run.results)


def run_report(ctx: E2EContext, opts: Options) -> str:
    """Throughput, speedup and perplexity of the measured builds, as one table (also saved as report.txt).
    Without a results file it is rebuilt from the measure output given with --log."""
    from .report import render_report
    if not ctx.run.results.exists():
        from .report_from_log import results_from_log
        results_from_log(require(opts.log, "--log (measure's output) when there is no results file", "report"),
                         ctx.run.results, model=ctx.model_config["file"], pp=opts.pp, tg=opts.tg)
    text = render_report(ctx.run.results, opts.baseline)
    ctx.run.report.write_text(text + "\n")
    print(text)
    return text


@dataclass(frozen=True)
class Step:
    name: str
    run: Callable[[E2EContext, Options], object]
    on_box: bool = True                                         # runs on the box (False: drives it over SSH)
    outputs: Callable[[E2EContext], list] = lambda ctx: []      # artifacts that mean "already done"


STEPS = (
    Step("kernel_set", run_kernel_set, outputs=lambda ctx: [ctx.run.manifest]),
    Step("deploy", run_deploy, on_box=False),
    Step("measure", run_measure, outputs=lambda ctx: [ctx.run.results]),
    Step("report", run_report),
)


def selected(opts: Options) -> list[Step]:
    known = [s.name for s in STEPS]
    unknown = [n for n in opts.steps or [] if n not in known]
    if unknown:
        raise SystemExit(f"evaluate has no step(s) {unknown}; steps: {known}")
    return [s for s in STEPS if opts.steps is None or s.name in opts.steps]


def run_here(ctx: E2EContext, opts: Options, steps: list[Step]) -> int:
    """Run `steps` in this process, in order, stopping at the first failure."""
    for step in steps:
        done = step.outputs(ctx)
        if done and all(p.exists() for p in done) and not ctx.force:
            print(f"[e2e] {step.name}: skipped, {', '.join(map(str, done))} already there (--force to redo)")
            continue
        print(f"[e2e] {step.name}")
        try:
            step.run(ctx, opts)
        except StepFailed as e:
            print(f"[e2e] {step.name} failed: {e}", flush=True)
            return 1
    return 0


def run_steps(ctx: E2EContext, opts: Options) -> int:
    """After each agent run: compile the kernels, splice them into the model, measure."""
    steps = selected(opts)
    ctx.run.root.mkdir(parents=True, exist_ok=True)
    if opts.here:
        if opts.steps is None:
            steps = [s for s in steps if s.on_box]
        if any(not s.on_box for s in steps):
            raise SystemExit("--here runs steps on the box; deploy drives the box, run it from the machine that launches it")
        return run_here(ctx, opts, steps)

    from .common.box import get_box
    from .remote import fetch, run_on_box
    try:
        if opts.fetch:
            fetch(get_box(box_label(ctx, opts)), ctx)
            return 0
        box = None
        for step in steps:
            if not step.on_box:
                print(f"[e2e] {step.name}")
                box = step.run(ctx, opts)
        on_box = [s.name for s in steps if s.on_box]
        if not on_box:
            return 0
        return run_on_box(box or get_box(box_label(ctx, opts)), ctx, opts, on_box)
    except StepFailed as e:
        print(f"[e2e] failed: {e}", flush=True)
        return 1
