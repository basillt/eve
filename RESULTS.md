# Results: are MoE expert-usage patterns personal?

Model: `allenai/OLMoE-1B-7B-0125-Instruct` (16 MoE layers × 64 experts, top-8), Colab T4.
Template tokens (chat template / system prompt) are excluded from all counts (`--skip-template`).

## Experiment 5 — 32 real users (WildChat-1M)

- Data: `allenai/WildChat-1M` (ODC-BY). First user turn of each conversation, English, 15–1500 chars,
  grouped by `hashed_ip`. 32 users × 45 prompts = 1440 prompts, 122k routed tokens (prompt + 24 generated).
- Split: per user, the first 70% of prompts (by time) build the core; the last 30% test it.
- **Personal core**: the user's most-used experts per layer.
  **Population core**: leave-one-out — built from all *other* users' training prompts (per-user normalized).
  **Oracle**: best possible static core chosen on the test prompts themselves.

### Static core hit rate (share of expert activations served from RAM)

| RAM budget | personal | population | oracle | random | users where personal wins |
|---|---|---|---|---|---|
| 10% | 25.2% ± 1.2 | 17.6% ± 1.2 | 26.7% | 10% | 31/32 |
| 20% | 41.5% ± 1.4 | 32.8% ± 1.5 | 43.9% | 20% | 31/32 |
| 30% | 52.9% ± 1.3 | 44.0% ± 1.4 | 55.6% | 30% | 31/32 |
| 50% | 72.8% ± 1.0 | 64.5% ± 1.1 | 75.5% | 50% | 31/32 |

(mean over users ± standard error)

- Experts per layer needed for 90% coverage: personal 47.4/64, population 52.2/64, oracle 45.4/64.
- Cross-user check at 20%: a user's core on **other** users' prompts hits 27.6%, vs 41.5% on the
  user's own later prompts.

**Finding 1 (robust):** expert routing is user-specific. A core built from ~31 of a user's own
prompts beats a core built from 31 other users for 31/32 users, and gets within ~2.5 points of the oracle.

### Cache simulation (same test prompts, token by token)

| policy | 20% cache (13/64) | 35% cache (22/64) |
|---|---|---|
| static population core | 32.8% | 49.0% |
| static personal core | 41.6% | 58.0% |
| LRU (request-level caching, generic warm start) | **44.6%** | 62.9% |
| LRU warm-started from personal core | 44.6% | 62.9% |
| half pinned personal + half LRU | 43.4% | **63.6%** |

- Warm-starting LRU from the personal core: +0.0 points (LRU forgets the warm start within a few tokens).
- Pinned personal + LRU vs LRU: −1.2 points at 20% (better for 10/32 users), +0.7 at 35% (better for 21/32).

**Finding 2 (negative, important):** as a *cache replacement policy*, a static personal core does **not**
beat plain LRU. Short-range temporal locality inside a request is stronger than long-range per-user
preference. A hybrid gives at most a small gain at larger budgets.

## What this means for the project

- The claim "a personal core saves RAM better than existing expert caching" is **not supported**.
  Prior work (MoE-Infinity, FineMoE) uses request-level / fine-grained activation tracing and would
  match or beat a static personal core on hit rate.
- What *is* supported: per-user routing statistics are stable and specific. That signal is useful where
  a cache cannot adapt, for example:
  - which experts to preload at app start (before any token is seen);
  - per-user mixed precision (tested in Experiment 6 below: the ranking works, the scheme loses to uniform 4-bit);
  - deciding which experts get LoRA adapters during nightly training.
- Engine default: `ExpertStore` already implements "pinned hot core + LRU"; the pinned share should be
  small (≤ half of the cache) and the LRU part does most of the work.

## Experiment 6 — personal mixed precision (same 32 users)

Question: a cache can adapt, quantization cannot (it is done once, offline). So does knowing a
user's experts help decide **which experts keep higher precision**?

- Script: `quant_eval.py`. Each user's later prompts (448 sequences in total) plus a 32-token greedy
  reply from the full fp16 model. Quality = KL(full fp16 || quantized) per token, on the reply tokens
  (template tokens skipped). Round-to-nearest, asymmetric, group 64 (+0.5 bit/weight for scales).
- hi experts at 8-bit, lo experts at 3-bit. The hi set per layer is chosen by four rules with
  identical memory. Routing check vs. the profiling run: correlation 0.978.

| config | bits/weight | KL on reply tokens | top-1 agreement |
|---|---|---|---|
| all 8-bit | 8.50 | 0.0013 | 98.8% |
| all 4-bit | 4.50 | 0.0379 | 91.7% |
| all 3-bit | 3.50 | 0.1446 | 84.6% |
| 12.5% hi — personal | 4.12 | 0.1029 | 86.9% |
| 12.5% hi — oracle | 4.12 | 0.1026 | 86.7% |
| 12.5% hi — population | 4.12 | 0.1192 | 85.9% |
| 12.5% hi — random | 4.12 | 0.1266 | 85.4% |
| 25% hi — personal | 4.75 | 0.0815 | 88.3% |
| 25% hi — oracle | 4.75 | 0.0779 | 88.4% |
| 25% hi — population | 4.75 | 0.0976 | 87.3% |
| 25% hi — random | 4.75 | 0.1091 | 86.6% |

Share of the 3-bit quality loss removed: personal 29% / 44% (12.5% / 25% budget), population 18% / 33%,
random 12% / 24%, oracle 29% / 47%. Personal beats population for **31/32 users** at both budgets.

**Finding 3 (positive):** the per-user ranking transfers to real output quality. Choosing the protected
experts from ~31 of the user's own prompts is about as good as the oracle and removes ~1.6× more of the
quantization loss than a population ranking at 12.5% budget.

**Finding 4 (negative, important):** this two-level scheme is not competitive with plain uniform 4-bit.
Uniform 4-bit (4.50 bits) has KL 0.038; the best personal mix (4.75 bits) has 0.081. With RTN the error
roughly quadruples per bit removed, so the experts left at 3-bit dominate, and usage is not concentrated
enough (top 25% of experts cover only ≈ 47% of a user's later activations) for a few protected experts to compensate. A rough
estimate says a 3/5-bit or 4/6-bit personal mix at equal memory would also lose to uniform. So
"personal mixed precision saves memory" is **not supported** with simple RTN quantization.

### What is left that is solid

1. Expert usage is user-specific and stable over time (Exp. 5, 31/32 users).
2. A small amount of a user's history predicts their important experts almost as well as an oracle,
   both for coverage (Exp. 5) and for quantization sensitivity (Exp. 6).
3. Neither a static personal cache nor personal 2-level precision beats the strong simple baseline
   (LRU caching; uniform 4-bit). Any product claim must be built on top of those baselines.

## Earlier experiments (summary)

| # | setup | result |
|---|---|---|
| 1 | Granite 3.1 1B-a400m (32 experts), personal vs generic | negative: oracle needs ~75% of experts for 90% |
| 2 | Qwen2.5-1.5B, top-10% neurons (ReLU-less SiLU MLP) | weak: +3.1 points |
| 3–4 | OLMoE, one person vs broad mixed prompts | personal 44.9% vs generic 34.0% at 20%; other person 26.7% |

Experiments 3–4 used a single synthetic "person"; experiment 5 replaces that with 32 real users.

## Reproduce

```
pip install datasets
python wildchat_users.py --users 32 --per-user 45 --min-prompts 45 --scan 400000
python profile_experts.py --model allenai/OLMoE-1B-7B-0125-Instruct --prompts prompts/wildchat_users.jsonl \
  --out results_users/all.npz --trace-out results_users/trace.npz --skip-template \
  --max-new-tokens 24 --max-prompt-tokens 256 --gpu-mem 13GiB
python analyze_users.py --counts results_users/all.npz --prompts prompts/wildchat_users.jsonl \
  --trace results_users/trace.npz --outdir results_users
python quant_eval.py --model allenai/OLMoE-1B-7B-0125-Instruct --prompts prompts/wildchat_users.jsonl \
  --counts results_users/all.npz --outdir results_quant      # ~100 min on a T4
```
About 22 minutes for profiling on a T4. User selection is random; numbers will vary slightly between runs.
