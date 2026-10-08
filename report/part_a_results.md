# Part A: intent model results

Data: MASSIVE v1.1, ar-SA + en-US, official train/dev/test (23,028 / 4,066 / 5,948 combined rows).
Test was evaluated once per frozen finalist. Dev was used for all selection.

## Fine-tuned models (full test, n = 5,948)

| | XLM-R base | E5-small (shipped) |
|---|---|---|
| Parameters | 278.1M | 117.7M |
| Size on disk | 1,129 MB | 488 MB |
| Accuracy | 0.8485 [0.837, 0.860] | 0.8384 [0.827, 0.849] |
| Macro-F1 | 0.8042 [0.785, 0.826] | 0.8056 [0.786, 0.825] |
| ar-SA acc / macro-F1 | 0.8130 / 0.7585 | 0.8020 / 0.7618 |
| en-US acc / macro-F1 | 0.8840 / 0.8464 | 0.8749 / 0.8496 |
| Latency CPU p50 / p95 (batch 1) | 29.5 / 40.6 ms | 12.0 / 15.9 ms |
| Latency GPU p50 / p95 (batch 1) | 6.8 / 7.7 ms | 6.6 / 7.5 ms |

CIs: cluster bootstrap over utterance ids (ar/en rows of one id are parallel), 1,000 resamples.
Paired E5 minus XLM-R: accuracy -1.0 pt [-1.8, -0.2] (McNemar p = 0.008);
macro-F1 +0.14 pt [-1.2, +1.4]. Per language macro-F1: ar +0.33 [-1.51, +2.24], en +0.32 [-1.34, +2.31].

## LLM zero-shot baseline vs fine-tuned (same 1,000 test rows = 500 ids x 2 locales)

LLM: allam-2-7b via Groq free tier, prompt zero_shot_v1 (60 labels, no descriptions, no examples),
temperature 0, prompt frozen before the test run. Invalid outputs count as wrong.

| | accuracy | macro-F1 | ar-SA acc | en-US acc |
|---|---|---|---|---|
| XLM-R | 0.858 | 0.853 | 0.832 | 0.884 |
| E5-small | 0.830 | 0.830 | 0.814 | 0.846 |
| allam-2-7b zero-shot | 0.563 | 0.619 | 0.516 | 0.610 |

Fine-tuned minus LLM: +26.7 (E5) and +29.5 (XLM-R) accuracy points, paired CIs exclude 0.
LLM invalid-output rate 2.7%, fuzzy-parsed 1.0%, 0 truncations.
Note: on this sample XLM-R leads E5 by 2.8 accuracy points, larger than on the full test (1.0);
the full test is the better estimate for that comparison.

## Latency and cost

| | latency | tokens / request | cost / request |
|---|---|---|---|
| E5-small, local CPU | p50 12.0 ms, p95 15.9 ms | n/a | no per-request fee; hardware cost not measured |
| XLM-R, local CPU | p50 29.5 ms, p95 40.6 ms | n/a | same |
| allam-2-7b, Groq API | p50 379 ms, p95 1,902 ms | 447 in / 5.9 out | not computed: no list price available; free-tier actual cost $0 |

Caveats: GPU latencies are launch-bound at batch 1 and do not separate the models. LLM latency
is measured from Cairo over the network, includes the provider's fast hardware, and excludes
rate-limit backoff (908 of 1,000 calls were retried, 3,371 s total). The free tier is not a
production option. Local latency excludes model loading; tokenization is included.

## Decision

Ship E5-small (seed 42, results/runs/e5-small-e15). The pre-registered rule was: ship the small model
if its macro-F1 is within 1.5 points of XLM-R on both languages. It is +0.33 on both, so it passes.
The rule is met on point estimates; the 95% lower bound for ar-SA is -1.51, at the margin.
XLM-R keeps a small but statistically real accuracy edge (about 1 point). E5 is 2.5x faster
on CPU and 2.3x smaller, which matters for a CPU-only Docker service. Both fine-tuned models
beat the zero-shot LLM by a wide margin. Seeds 43/44 were dev-only and not used for selection.

## Error analysis

- Arabic is about 6-7 accuracy points below English for both architectures and all three E5 seeds.
  Mean tokens per utterance are equal (10.4 ar vs 10.1 en), so tokenization length is not the cause.
  104 ar-SA test rows (3.5%) contain no Arabic letters and score lower (about 74%), but removing them
  closes only about 0.3 points. Translation noise is a plausible cause, untested.
- Errors concentrate in general_quirky and qa_*: 15.4% of test rows but 38% (XLM-R) / 36% (E5) of errors.
  Persistent pairs: qa_factoid <-> general_quirky, calendar_set / calendar_query / calendar_remove,
  transport_ticket -> transport_query (Arabic), play_radio -> play_music.
- The LLM's largest confusion is calendar_set -> alarm_set (37 errors), a label-definition problem
  that descriptions or examples could reduce; not tested.

## Limitations

XLM-R is a single seed; E5 dev variance comes from 3 seeds; one training recipe per model.
Per-intent supports are small (about 25 dev examples for rare intents). Dev loss rises while
accuracy plateaus, so confidences are overconfident; the router will need temperature scaling.
Training environment: Kaggle (torch 2.11.0+cu128, transformers 5.16.1, sklearn 1.6.1); local:
torch 2.14.1+cu126, transformers 5.19.0, sklearn 1.9.1. A local CPU re-evaluation reproduced Kaggle dev numbers.