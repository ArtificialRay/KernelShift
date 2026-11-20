# kvr: one kernel against llama.cpp's own paths

`kvr` (kernel vs repack) tests one kernel at a time. It asks: for one matrix multiplication, is the agent's
kernel faster than the code llama.cpp would run for it?

It runs the same multiplication three ways, on one thread, with the same weights and the same input. It does
this for several M, the number of input rows: M = 1 is like generating one token, a larger M is like reading a prompt.

## The three ways

- **repack**: ggml's fastest built-in code. When the model loads, ggml rearranges ("repacks") the weights into
  a layout its fast code reads well (`repack.cpp`; on Graviton4 the i8mm `8x8` kernels). Stock `llama-bench`
  uses this.
- **norepack**: the same ggml call, but the weights stay in their normal layout, so ggml uses its plain code.
  The norepack build uses this, and so does the `baseline-llamacpp` baseline.
- **agent**: the agent's `.so`, called exactly like the hook in `e2e/override` calls it. First the input is
  converted from f32 to bf16 (`ggml_cpu_fp32_to_bf16`), then the kernel computes the `[M,N]` result. The
  conversion is counted in the agent's time.

Each ggml call also converts its input into its own quantized format (Q8_K / Q8_0), and that is counted too,
so all three times are for a complete multiplication. After a few warmup calls, the three ways take turns
(repack, norepack, agent, repeat), and the fastest and the median time of each are kept.

## How the harness works

1. Generate Gaussian f32 weights for the shape and quantize them with `ggml_quantize_chunk` (or read
   `--weights raw_rows.bin`). Activations are N(0, 1) (or `--act x.npy|x.bin`). Generation is not timed;
   ggml's quantizers run at about 1 µs per element per core, so the 635M-element q6_K lm_head takes about
   a minute on 16 cores.
2. Allocate the weight twice, once in the CPU_REPACK buffer and once in a plain buffer. If ggml has no repack
   kernel for the type, N and CPU (repack covers q4_K, q5_K, q6_K with N % 8 == 0 and q8_0 with N % 4 == 0),
   the repack path is skipped: `repack_used` is `false` and the repack times are `null`.
3. Time the three paths for each M, interleaved.
4. Check the results: compare the agent's output with the repack output (SQNR) and, on a sample of columns,
   each path with an fp64 reference computed from the dequantized weights.

`ARMBENCH_OVERRIDES` is unset for the run, so the override hook cannot take over the plain-buffer path.

## What it produces

One JSON line per shape and M, in `kvr.jsonl`:

- `shape`, `type`, `N`, `K`, `M`: what was timed.
- `t_repack_min_us` / `t_repack_med_us`, `t_norepack_*`, `t_agent_*`: best and median time per path in µs.
- `t_bf16_convert_min_us`: the part of the agent time spent converting the activations.
- `agent_vs_repack` = `t_repack_min / t_agent_min`, and `agent_vs_norepack`, `repack_vs_norepack` likewise.
  Above 1, the first path is faster.
- `sqnr_db`, `max_abs_agent`: the agent's output against the repack output.
- `sqnr_norepack_vs_repack_db`: the two ggml paths against each other, about 130 dB or more since both
  quantize the activations the same way.
- `sqnr_fp64ref_*_db`: each path against the fp64 reference.
- `repack_used`, `repack_kernel`: whether ggml repacked the weight, and which kernel it chose (for example
  `q4_K_8x8`).

`e2e evaluate` also prints these as a table in `report.txt`.

## Running it

Through `e2e`, which builds the harness against `build-stock` on the box and runs every mul_mat kernel of the
run's `manifest.json`:

```
python -m e2e evaluate --model qwen3.5-4b --steps kvr,report --kvr-m 1 8 64 512
```

By hand, on the box:

```sh
./build.sh ~/llama.cpp-e2e ~/llama.cpp-e2e/build-stock          # -> ./bench_repack (-O3 -mcpu=native)

# one shape
./bench_repack --type q4_K --N 2560 --K 9216 --M 1,2,4,8,16,32,128,512 \
    --so gemm_ggml_q4_K_n2560_k9216.so --symbol kernelshift_entry_gemm --abi entry_rows

# every kernel of a manifest: one line per shape and M, plus a summary table on stderr
./run_all.sh manifest.json kvr.jsonl
```

`build.sh` links `build-stock`'s `libggml`, `libggml-cpu` and `libggml-base` (static or shared), which is the
build with `GGML_CPU_REPACK=ON`; `build-agent` and `build-norepack` have it off and no repack buffer.

Options of `bench_repack` (pass several through `BENCH_ARGS` of `run_all.sh`):

- `--reps N` (default 20) timed rounds, `--warmup N` (3), `--max-sec S` (4) time budget per M.
- `--flush-mb N` streams through N MB before every timed call. The weights of a shape are small (1.5 MB for
  q4_K 1024x2560) and stay in L2 between calls, while real decode reads them from DRAM; `--flush-mb 256`
  (more than L3) gives the cold-cache comparison.
- `--ref-rows N` (64): columns checked against the fp64 reference. `--gen-threads N`: threads for weight generation.

Environment variables of `run_all.sh`: `MS` (M values, default `1,2,4,8,16,32,128,512`), `BIG_N` and `BIG_MS`
(shapes with N above `BIG_N`, default 100000, use `BIG_MS`, default `1,2,4,8`; this covers the lm_head),
`PIN_CPU` (CPU for `taskset`, default 2, empty disables pinning).

## Reading the numbers

- The repack path does rows in groups of 4 with gemm and the leftover rows with gemv, so its time steps at
  M = 4k + r.
- `q4_K_8x8` and `q8_0_4x8` have an SVE path for 256-bit SVE only. Graviton4 (128-bit) takes the NEON i8mm
  path, Graviton3 takes the SVE one.
