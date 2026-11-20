#!/usr/bin/env python3
"""Bring up the end-to-end measurement tree on the box labelled `label`.

The box is found through provisioning/ (e2e.common.box); with `provision=True` it is created first.
config/dataset_builds.json's "llama.cpp-e2e" entry is run through ensure_dataset_ready (pinned
clone, stock tools build); the config/e2e_models.json GGUF is then fetched into ~/models on the box.
With agent_build it also derives the override-enabled "agent" build via e2e/build_agent.sh, which is
rsynced to the box together with e2e/override/.

  deploy("e2e-qwen3.5-4b-sve2", "qwen3.5-4b", agent_build=True)    # or: python -m e2e evaluate --steps deploy ...
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from e2e.common.box import get_box
from e2e.errors import StepFailed

REPO_ROOT = Path(__file__).resolve().parents[1]


def deploy(label: str, model: str | None = None, instance_type: str | None = None, agent_build: bool = False,
           on_demand: bool = False):
    """Bring up the end-to-end measurement tree on the box `label`, driving it over SSH, and return the box.

    model          config/e2e_models.json key whose GGUF to fetch into ~/models on the box
    instance_type  create the box with this type if none is registered under `label` (spends money);
                   None: a missing box is an error
    agent_build    also build the override-enabled "agent" tree (and the no-repack control)
    on_demand      create it on-demand instead of spot (a spot box can be reclaimed mid-run)
    """
    from provisioning.provision import ensure_dataset_ready
    box = get_box(label, create_instance_type=instance_type, on_demand=on_demand)
    print(f"[e2e] {label}: {box.user}@{box.host}")

    ensure_dataset_ready(box, "llama.cpp-e2e")

    if model:
        models = json.loads((REPO_ROOT / "config" / "e2e_models.json").read_text())
        if model not in models:
            raise StepFailed(f"unknown model {model!r}; known: {[k for k in models if not k.startswith('_')]}")
        m = models[model]
        cmd = (f"mkdir -p ~/models && cd ~/models && (test -f {m['file']} || (curl -sSL -C - -o {m['file']}.part "
               f"https://huggingface.co/{m['repo']}/resolve/main/{m['file']} && mv -f {m['file']}.part {m['file']})) && ls -la {m['file']}")
        print(f"[e2e] fetching {m['repo']}/{m['file']} ({m['size_gb']} GB) on the box")
        rc, out, err = box.run(cmd, timeout=3600)
        print(out.strip()[-300:])
        if rc != 0:
            raise StepFailed(f"model fetch failed: {err[-500:]}")

    if agent_build:
        box.run("mkdir -p ~/kernelshift-e2e/e2e", timeout=30)   # rsync only creates the last path component
        rc = subprocess.run(["rsync", "-az", "--exclude", "__pycache__", "build_agent.sh", "override",
                             f"{box.user}@{box.host}:kernelshift-e2e/e2e/",
                             "-e", f"ssh -i {box.key_file} -o StrictHostKeyChecking=accept-new"],
                            cwd=REPO_ROOT / "e2e").returncode
        if rc != 0:
            raise StepFailed("rsync of e2e/build_agent.sh and e2e/override failed")
        rc, out, err = box.run("bash ~/kernelshift-e2e/e2e/build_agent.sh ~/llama.cpp-e2e", timeout=1800)
        print(out[-2000:]); print(err[-2000:], file=sys.stderr)
        if rc != 0:
            raise StepFailed("agent build failed")
    print("[e2e] ready")
    return box
