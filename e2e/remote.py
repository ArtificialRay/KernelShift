"""Run the box-side steps of `evaluate` on the Graviton box and bring the run directory back.

The launcher (the machine `python -m e2e evaluate` is typed on) takes no part in the work: it syncs the
repo to the box, starts `python -m e2e evaluate ... --here` there detached (nohup, so a dropped SSH
session does not stop it), follows its log, and rsyncs `e2e-runs/<run>/` back when it is done.
"""
from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import fields
from pathlib import Path

from .context import REPO, E2EContext, Options
from .errors import StepFailed

BOX_REPO = "~/arm-bench"
POLL_SECONDS = 30
LAUNCHER_ONLY = {"steps", "here", "fetch", "root", "provision", "instance_type", "label", "log"}


def box_args(ctx: E2EContext, opts: Options, steps: list[str]) -> list[str]:
    """The command line of the box-side run: every option that differs from its default, `--here`
    and the steps to run there."""
    # the box keeps the run under the same directory name, so that fetch finds it
    args = ["--model", ctx.model, "--isa", ctx.isa, "--run-dir", f"e2e-runs/{ctx.run.root.name}"]
    if ctx.force:
        args.append("--force")
    default = Options()
    for f in fields(Options):
        value = getattr(opts, f.name)
        if f.name in LAUNCHER_ONLY or value == getattr(default, f.name):
            continue
        flag = "--" + f.name.replace("_", "-")
        if isinstance(value, bool):
            args.append(flag)
        elif isinstance(value, list):
            args += [flag, *map(str, value)]
        else:
            args += [flag, str(value)]
    return args + ["--here", "--steps", ",".join(steps)]


def run_on_box(box, ctx: E2EContext, opts: Options, steps: list[str]) -> int:
    """Sync the repo to the box, run `steps` there and fetch the run directory. The exit code of the run."""
    from provisioning.provision import RSYNC_ALLOWLIST
    run = f"{BOX_REPO}/e2e-runs/{ctx.run.root.name}"
    print(f"[e2e] syncing {REPO} -> {box.host}:{BOX_REPO}")
    try:
        box.rsync_to(str(REPO), BOX_REPO, paths=RSYNC_ALLOWLIST)
    except subprocess.CalledProcessError as e:
        raise StepFailed(f"rsync to the box failed: {(e.stderr or b'').decode()[-300:]}")

    command = f"cd {BOX_REPO} && {box.python} -m e2e evaluate " + " ".join(map(shlex.quote, box_args(ctx, opts, steps)))
    # setsid + every stream redirected: otherwise ssh waits for the detached run to close its stdin/stdout
    box.run(f"mkdir -p {run} && rm -f {run}/evaluate.done && "
            f"setsid nohup bash -c {shlex.quote(f'{command} > {run}/evaluate.log 2>&1; echo $? > {run}/evaluate.done')} "
            f"</dev/null >/dev/null 2>&1 &", timeout=60)
    print(f"[e2e] running on {box.host}: {' '.join(steps)}   (log: {run}/evaluate.log)")
    try:
        code = follow(box, run)
    except KeyboardInterrupt:
        print(f"\n[e2e] stopped waiting; the run continues on the box. Pull its results with the same "
              f"command plus --fetch.")
        return 130
    fetch(box, ctx)
    return code


def follow(box, run: str) -> int:
    """Print the box-side log as it grows until the run's done-marker appears; its exit code."""
    offset, lost = 0, 0
    while True:
        rc, out = _poll(box, run, offset)
        if rc is None:
            lost += 1
            if lost > 20:
                raise StepFailed(f"lost the connection to {box.host} for {lost * POLL_SECONDS}s; "
                                 "the run may still be going, retry with --fetch")
        else:
            lost = 0
            text, code = out
            if text:
                print(text, end="", flush=True)
                offset += len(text.encode())
            if code is not None:
                return code
        time.sleep(POLL_SECONDS)


def _poll(box, run: str, offset: int):
    """(rc, (new log text, exit code or None)), or (None, None) when the box cannot be reached."""
    try:
        rc, text, _ = box.run(f"tail -c +{offset + 1} {run}/evaluate.log 2>/dev/null", timeout=60)
        rc2, done, _ = box.run(f"cat {run}/evaluate.done 2>/dev/null", timeout=60)
    except subprocess.TimeoutExpired:
        return None, None
    if rc == 255 or rc2 == 255:
        return None, None
    return 0, (text, int(done.strip()) if done.strip() else None)


def fetch(box, ctx: E2EContext) -> None:
    """Copy `e2e-runs/<run>/` of the box into the local run directory (the compile scratch is left out)."""
    ctx.run.root.mkdir(parents=True, exist_ok=True)
    box.rsync_from(f"{BOX_REPO}/e2e-runs/{ctx.run.root.name}/", str(ctx.run.root), excludes=["build/"])
    print(f"[e2e] run directory -> {ctx.run.root}")
