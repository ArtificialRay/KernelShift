# End-to-end: agent kernels inside llama.cpp

The rest of the benchmark measures one kernel at a time. `e2e` answers the follow-up question: take the
kernels an agent wrote for a real model's matmuls, splice them into llama.cpp, and measure tokens/s and
perplexity.

## Before you run it

- **Definitions** tagged `e2e:<model>` in `bench-trace/` (the model is a key of `config/e2e_models.json`).
  For a new model, `plugins/llamacpp/` (`kgen discover`) and the `gen-definition` skill make them.
- **Agent solutions** under `bench-trace/solutions/llama.cpp/<author>/gemm/`: run the agents with
  `test_scripts/bench_fleet.py --dataset llama.cpp --definitions "<names>" --sync-solutions ...`.
- **A Graviton box** registered in `provisioning/eval_config.json` under the label `e2e-<model>-<isa>`, use `--label` to pick the exact instance you like to run the e2e experiment. Or, you could let `evaluate` create it with `--provision`

## Usage

```
python -m e2e evaluate --model qwen3.5-4b --author AUTHOR --rows-abi --agent-build [--provision] [--perplexity FILE]
```

One command runs everything. The work happens on the box: `evaluate` syncs the repo there, starts the
steps detached (a dropped SSH session does not stop them), follows their log, and copies
`e2e-runs/llamacpp-<model>-<isa>/` back when they finish. Ctrl-C only stops the following; pull the
results later with the same command plus `--fetch`. `--perplexity FILE` is a path on the box.

| Step | Runs | What it does | Main options |
|---|---|---|---|
| `kernel_set` | box | Compiles the author's submitted kernels into shared objects (clang++-18, aarch64) and writes the manifest that tells llama.cpp which matmuls to replace. | `--author` (required), `--kernel all\|NAME`, `--rows-abi`, `--reference-author` with `--expect-changed`, `--expect-same`, `--expect-count`, `--forbid` |
| `deploy` | launcher, over SSH | Builds the pinned llama.cpp tools tree on the box, downloads the model, and with `--agent-build` builds the variant with the replacement hook. | `--label`, `--provision`, `--instance-type`, `--agent-build` |
| `measure` | box | llama-bench for each build, interleaved and repeated, and llama-perplexity. Fails if `build-agent` exists but the manifest is missing or empty. | `--perplexity FILE`, `--threads`, `--pp`, `--tg`, `--reps`, `--fa` |
| `report` | box | Prints throughput, speedup and perplexity of the builds as one table, saved as `report.txt`. If the results file is missing it is rebuilt from `--log`, the output `measure` printed. | `--baseline`, `--log` |

`--steps a,b` runs only some of them; a step whose result is already in the run directory is skipped, you can use `--force` to redo the whole process.

`--here` runs the steps in the current process instead of launching them on the box. you only need it after logging in to the box to redo one step, e.g.
`python -m e2e evaluate --model qwen3.5-4b --here --steps measure`.

- `--kernel all` (default) uses every submitted definition tagged `e2e:<model>`; `--kernel NAME` just one kernel, which the `--author` must have submitted.
- `--rows-abi` lets the hook split a matmul's rows across ggml's threads.
- The guards compare the set with a previous author's: `--reference-author PREV --expect-changed DEF...`
  fails if a kernel you meant to change is byte-identical to PREV's.
- Control run: `--author reference-scalar --rows-abi` builds the scalar set, which should reproduce stock perplexity.

The run directory holds `manifest.json` and the `.so` files, `e2e_results.json` (every sample, the
summary, perplexity, host, date and llama.cpp commit), `hook_agent.log` (which kernels the hook replaced,
with which ABI and how each call was split), `report.txt` and `evaluate.log` (what the box printed).

Tearing the box down stays with `provisioning/provision.py --teardown --label ...`.

## What is compared

Three builds: 
- **stock**: llamacpp default setting on this e2e model 
- **norepack**: llamacpp with ggml's repack and KleidiAI paths off, enable with `--build-norepack`
- **agent**: the norepack configuration plus the override hook, which replaces only the matmuls listed in the manifest with the agent's kernels, enable with `--build-agent`

Details of the hook: `e2e/override/README.md`.
