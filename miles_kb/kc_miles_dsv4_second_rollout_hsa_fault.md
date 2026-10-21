# Knowledge Candidate: DSV4 32-GPU RL dies entering the SECOND rollout, right after the first weight sync

> Proposed knowledge. Lives in `exp/<run_id>/knowledge_candidates/`. NOT yet in the official
> knowledge_base — promotion is human-gated.

- id: kc_miles_dsv4_second_rollout_hsa_fault
- created_utc: "2026-10-21T07:40:00Z"
- related_case_ids: [dsv4_4node_cp1tp8pp4ep8, dsv4_4node_cp8tp1pp4ep8, dsv4_4node_cp1tp8pp4ep8_moetriton]
- confidence: medium (signature is reproducible and tightly localised; the faulting kernel is not yet root-caused)

## Symptom

A 4 × 8 MI308X DeepSeek-V4-Flash colocate RL run completes rollout 0, the training step, and the
first weight sync — and then dies within **27–41 log lines** of `Timer update_weights end`:

```
Queue error: HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception. code: 0x1016
GPU index: <2..7>, kernel: void at::native::_assert_async_cuda_kernel<bool>
[rank N] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=265, OpType=ALLREDUCE, ...)
```

SGLang engines then fail their HTTP health checks one by one
(`"No sglang scheduler processes found for CUDA coredump"`), surviving ranks hang in the all-reduce
until the watchdog fires, and the job reports `0/N steps completed`. `rollout/rewards` and
`train_rollout_kl` are never emitted for the step in progress.

## Root cause

Not yet identified. What IS established is the **fault is confined to the transition from the first
weight sync into the second rollout**, and is independent of every axis tested:

| varied | values tested | fault |
|---|---|---|
| context parallel | CP=1 and CP=8 | both |
| tensor parallel | TP=8 and TP=1 | both |
| SGLang MoE runner | AUTO (=AITER) and `triton` | both |
| distributed timeout | 10 min and 60 min | both |

Fault position relative to the first weight sync, across three independent runs:

| run | `Timer update_weights end` | first `HSA_STATUS_ERROR` | gap | sync cost |
|---|---|---|---|---|
| cp1tp8pp4ep8 (AITER) | line 5029 | line 5070 | 41 | 824.4 s |
| cp8tp1pp4ep8 (AITER) | line 5092 | line 5128 | 36 | 965.5 s |
| cp1tp8pp4ep8 + triton MoE | line 4970 | line 4997 | 27 | 790.4 s |

**`at::native::_assert_async_cuda_kernel<bool>` is PyTorch's device-side assertion**; when it fails
it traps and aborts the HSA queue, so the RCCL watchdog timeouts and the engine deaths are
downstream, not the cause. Raising `distributed_timeout_minutes` from 10 to 60 reduced the cascade
(watchdog hits 64 → 4) but did not prevent the fault — it only lengthened the teardown.

Leading hypothesis, unconfirmed: SGLang's in-place `update_weights_from_tensor` leaves part of the
model in an inconsistent quantisation state, so the next forward produces non-finite logits and the
sampler's device assert fires. The weight sync logs **168 × `Skip dequant fp8 wo_a`**, i.e. FP8
dequantisation of the attention output projection is being skipped during the update.

## Why this was never seen before

**It had never been exercised.** Across all 19 historical 4-node logs in this run directory, the
only rollout id that ever appears is `rollout_id=0` — `rollout_id=1` appears nowhere. Four of those
runs requested `--num-rollout 3000` and still stopped at rollout 0, with **no terminal error in the
log** (they were stopped by the operator). The maximum steps ever completed historically is 1.

So "it used to work" means "one rollout used to complete". The second rollout — and therefore the
weight-sync→resume-generation path — is new territory, which is exactly why three configurations
that differ in CP, TP and MoE backend all break at the same place.

## Diagnosis

```bash
# Is the fault downstream of the first weight sync? (expect a gap of tens of lines)
grep -an "Timer update_weights end" train.log | head -1
grep -an "HSA_STATUS_ERROR\|Queue error"  train.log | head -1

# Which kernel trapped, and on which GPUs
grep -aoE "GPU index: [0-9]+, kernel: [A-Za-z:_<> ]{0,60}" train.log | sort -u

# Rule the cascade out as a cause: watchdog must come AFTER the HSA error
grep -an "Watchdog caught collective" train.log | head -1

# Has a second rollout ever started?
grep -aohE "rollout_id=[0-9]+" train.log | sort -u

# The dequant-skip signal during the sync
grep -aoE "Skip dequant fp8 [a-z_]+" train.log | sort | uniq -c
```

## Fix / workaround

None known. Axes already ruled out (do not re-test): CP size, TP size, SGLang MoE runner backend,
collective timeout, and any difference against the in-image reference
`/opt/miles-rocm/run_full_6node.sh` — a full config diff showed the effective compute configuration
(parallel layout, 22 runtime env vars, all SGLang/train args, all 23 launcher flags) is identical;
only the rollout batch size differs (6 vs 4).

Suggested next probes, in order:
1. Instrument the first forward after `update_weights_from_tensor` for non-finite logits — this
   either confirms or kills the dequant hypothesis cheaply.
2. Inspect the `wo_a` tensor's scale/quant state on both sides of the sync.
3. Only then consider `CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1` — note that GPU core dumps on this
   host class have previously written ~45 GiB per card and filled an 879 GB filesystem, so sweep
   `gpucore*` and keep `AMD_GPU_COREDUMP=0` set for every other run.

## Regression test

A 4-node DSV4 case with `--num-rollout 2` that asserts `rollout_id=1` is reached and that
`HSA_STATUS_ERROR`/`_assert_async` counts are zero. This is the first case in the suite that
exercises the second rollout at all; a 1-rollout case cannot catch it.

## Evidence

- `exp/miles_spec_20260817_1430/porting/runs_cp/dsv4_4node_cp1tp8pp4ep8_20260918_032335/train.log`
  (sync line 5029, fault line 5070, 32 × HSA, 7 × assert, 64 × watchdog)
- `.../runs_cp/dsv4_4node_cp8tp1pp4ep8_20260918_064605/train.log` (sync 5092, fault 5128, 25 × HSA)
- `.../runs_cp/dsv4_4node_cp1tp8pp4ep8_moetriton_20260918_080204/train.log`
  (sync 4970, fault 4997; `[aiter] fused_moe` count 0, confirming the bypass was active)
- Historical rollout ids and step counts: all 19 `exp/miles_spec_20260817_1430/porting/full4node*.log`
- Historical step-0 metrics for comparison: `train_rollout_kl` 0.00884 (2K seq) → 0.17741 / 0.17753 /
  0.19835 (16K seq) in `full4node_2k_tp8.log`, `full4node_16k_cp4tp2.log`, `full4node_16k.log`,
  `full4node_16k_cp8_clean2.log`

## Scope

- frameworks: [miles, sglang, megatron-lm]
- rocm: ["7.14"]
- gpu: [gfx942 / MI308X]
