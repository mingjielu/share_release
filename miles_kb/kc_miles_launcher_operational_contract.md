# Knowledge Candidate: Miles AMD launcher — operational contract for multi-node DSV4 runs

> Proposed knowledge. Lives in `exp/<run_id>/knowledge_candidates/`. NOT yet in the official
> knowledge_base — promotion is human-gated.

- id: kc_miles_launcher_operational_contract
- created_utc: "2026-10-21T07:40:00Z"
- related_case_ids: [dsv4_4node_cp1tp8pp4ep8, dsv4_4node_cp8tp1pp4ep8]
- confidence: high

> **Why this page**: `knowledge_base/02-components/` has a page for verl, vllm, sglang, swift,
> megatron-lm, aiter, ray, rccl, deepep, fused-moe, flash-attention and torch-fsdp2, but **none for
> Miles** — it is mentioned only in passing in `megatron-lm.md`. Every existing Miles-related
> candidate (~20 of them) documents a SGLang/ROCm/DSV4 symptom, not the framework's own operating
> rules. The items below are the Miles-level facts that cost the most time to rediscover.

## Symptom

Driving a multi-node Miles run from a host script fails in ways that never name Miles:

- `train.py: error: unrecognized arguments: --no-sequence-parallel` (~50 s in)
- `IndexError: list index out of range` inside `_parse_extra_env_vars`, with no indication that the
  `--extra-env-vars` payload was the problem
- `NotImplementedError: No pre-set parallel config for N GPUs` for any node count the launcher has
  not been taught
- SGLang refuses to size its KV cache, or OOMs, because the launcher's own SGLang defaults are for a
  much smaller model than DeepSeek-V4-Flash
- `Watchdog caught collective operation timeout ... Timeout(ms)=600000` during weight sync

## Root cause

Six independent properties of `scripts/amd/run_deepseek_v4.py` and `miles/utils/external_utils/command_utils.py`:

1. **The parallel layout is NOT taken from your script.** `_get_parallel_config` only returns
   layouts it has been explicitly taught, and raises `NotImplementedError` for anything else. In the
   images checked (`...128kcp-20260821` and `...cp8mem-20260825-fada`, launcher md5 `75ba10cd40ea`,
   byte-identical) it knows exactly two shapes:
   - `actor_num_nodes == 1`: TP=gpus, SP, PP=1, CP=1, EP=gpus, ETP=1
   - `total_gpus == 32` (4 × 8): **TP=8, SP, PP=4, layers 11+11+11+10, CP=1, EP=8, ETP=1**

   A script named `run_full_6node.sh` therefore cannot actually run 6 nodes on these images; when it
   works it is being given 4 hosts.

2. **`--extra-args` is appended LAST**, so it overrides `_get_parallel_config`. That is the only
   supported way to change the layout (e.g. `--tensor-model-parallel-size 1 --context-parallel-size 8
   --allgather-cp` for a CP=8 arm).

3. **`--extra-env-vars` is JSON, parsed by a fragile fallback.** `command_utils.py:209`:
   ```python
   try:    return json.loads(text)
   except ValueError:
           return {kv[0]: kv[1] for item in text.split(" ") if item.strip() != "" if (kv := item.split("=")) or True}
   ```
   Any token without `=` makes `kv[1]` raise `IndexError`. A JSON object contains spaces, so the
   moment the quoting is damaged the JSON silently degrades into this space-splitting path and dies
   with an error that names neither JSON nor the variable.

4. **The launcher's SGLang defaults target a smaller model**: `sglang_tp_size=4`, `sglang_ep_size=4`,
   `--sglang-mem-fraction-static 0.7`. DeepSeek-V4-Flash needs tp8/ep8 at ~0.43.

5. **`distributed_timeout_minutes` defaults to 10** (= `Timeout(ms)=600000`), but the first weight
   sync on this model measured **824.4 s / 965.5 s / 790.4 s** across three runs — a phase that
   reliably outlasts its own collective watchdog.

6. **Checkpoint paths are derived, not passed.** ref-load resolves to
   `{model_local_dir}/{model_name}_torch_dist`, and `--hf-checkpoint` overrides
   `{model_dir}/{model_name}` only for the HF tree. `model_name` is a `Literal` accepting just
   `DeepSeek-V4-Flash-FP8` and `DeepSeek-V4-Flash-FP8-4layer` — the older `DeepSeek-V4-Flash` is
   rejected at argument parsing.

## Diagnosis

```bash
# 1. What layouts does THIS image's launcher actually support?
docker run --rm --entrypoint bash <image> -lc \
  'grep -oE "total_gpus == [0-9]+|actor_num_nodes == [0-9]+" /opt/miles/scripts/amd/run_deepseek_v4.py | sort -u'

# 2. Confirm the layout that was really used (never trust the driver's banner)
grep -aoE "context-parallel size: [0-9]+|tensor-model-parallel size: [0-9]+" train.log | tail -2

# 3. Confirm the env payload survived the shell hops
grep -o "\-\-extra-env-vars '[^']*'" <run>/job.sh \
  | sed "s/--extra-env-vars '//; s/'$//" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))"

# 4. Confirm the timeout budget vs the measured phase cost
grep -aoE "distributed_timeout_minutes [. ]*[0-9]+" train.log | sort -u
grep -aoE "Timer update_weights end \(elapsed: [0-9.]+s\)" train.log
```

## Fix / workaround

- **Let an external driver own Ray**: set `MILES_SCRIPT_EXTERNAL_RAY=1` (read at
  `command_utils.py:125` via `get_bool_env_var`). Miles then skips its own `ray start` / `ray stop`
  / `pkill -9 ray`, so a host script can bring up Ray on non-default ports without having its
  cluster torn down. Without it, Miles' launcher starts Ray with no `--port` and lands on the
  default, colliding with any neighbouring container under `--network=host`.
- **Never build the job as `bash -lc "<job text>"`.** Write the whole command to a file on shared
  storage and run `docker exec <ctr> bash /path/job.sh`. The JSON in `--extra-env-vars` does not
  survive `ssh -> docker exec -> bash -lc`, and the failure surfaces as the `IndexError` above.
  (The in-image reference `/opt/miles-rocm/run_full_6node.sh` carries this same hazard.)
- **Override the SGLang block and the timeouts via `--extra-args`**:
  ```
  --sglang-tp-size 8 --sglang-dp-size 1 --sglang-ep-size 8 --sglang-mem-fraction-static 0.43
  --sglang-disable-cuda-graph --sglang-disable-custom-all-reduce
  --distributed-timeout-minutes 60 --rollout-health-check-timeout 1800 --sglang-watchdog-timeout 1800
  ```
- **Do not pass `--no-sequence-parallel`** — it does not exist in this build. Megatron disables
  sequence parallelism itself at TP=1 (`arguments.py:906`, "Disabling sequence parallelism because
  tensor model parallelism is disabled"), so leaving the launcher's `--sequence-parallel` in place
  is correct for a CP=8/TP=1 arm.
- **Point at the real assets** with `--model-name DeepSeek-V4-Flash-FP8 --model-dir <parent>
  --model-local-dir <parent> --hf-checkpoint <parent>/DeepSeek-V4-Flash-BF16`, and verify
  `<parent>/DeepSeek-V4-Flash-FP8_torch_dist/release/common.pt` exists before launching — a
  torch_dist tree missing that 24 KB bookkeeping file kills every train actor with
  `FileNotFoundError`, after ~20 minutes of startup.

## Metrics Miles emits (useful for grepping a run)

`train/train_rollout_kl`, `train/train_rollout_logprob_abs_diff`, `train/ess_ratio`,
`train/ppo_kl`, `train/pg_loss`, `train/pg_clipfrac`, `train/grad_norm`, `rollout/rewards`,
`perf/step_time`, `used_GB`. Step boundaries appear as `rollout_id=<N>, step=<M>`.

## Regression test

A 2-step, 4-node DSV4 case whose driver asserts, before training starts, that: the launcher's
supported-layout list contains the requested shape; the `--extra-env-vars` JSON round-trips through
`json.loads` from the generated job file; `distributed_timeout_minutes` exceeds the observed
`update_weights` elapsed time; and `release/common.pt` is present in the ref-load tree.

## Evidence

- Launcher layout table + `NotImplementedError`: `/opt/miles/scripts/amd/run_deepseek_v4.py`
  `_get_parallel_config` (md5 `75ba10cd40ea`, identical in both images checked)
- Fragile env parsing: `/opt/miles/miles/utils/external_utils/command_utils.py:209-213`
- External-Ray switch: `/opt/miles/miles/utils/external_utils/command_utils.py:125`
- `IndexError` in practice:
  `exp/miles_spec_20260817_1430/porting/runs_ref/dsv4_4node_cp1tp8pp4ep8_20260917_160000/train.log`
- `unrecognized arguments: --no-sequence-parallel`:
  `runs_cp/dsv4_4node_cp8tp1pp4ep8_20260918_045009/train.log:1418`
- Weight-sync cost vs 600 s budget: `Timer update_weights end (elapsed: 824.4s | 965.5s | 790.4s)`
  in the three `runs_cp/dsv4_4node_*` logs; `Timeout(ms)=600000` in the same logs
- Missing `common.pt`: `exp/miles_spec_20260817_1430/porting/recover/make_common_pt.py` and the
  `FileNotFoundError` it was written to repair

## Scope

- frameworks: [miles, sglang, megatron-lm]
- rocm: ["7.14"]
- gpu: [gfx942 / MI308X]
