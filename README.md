# agentfrailty

**Are agent failures independent?** Separating task frailty from within-run
dependence in local LLM agents.

> Status: **scaffold.** Nothing measured yet. The kill-test below is the gate
> that decides whether the study is viable on this hardware.

---

## The question

There is an open contradiction in the literature about how agents fail.

**Hazard declines.** Toby Ord proposed a constant-hazard model for agent
failure, then [publicly retracted it](https://www.tobyord.com/writing/hazard-rates-for-ai-agents-decline):
a Weibull refit of METR's data gives shape parameter *k* significantly below 1
for every model tested, clustering near 0.6. The longer a run survives, the
*less* likely it is to fail next.

**Hazard rises.** [Self-conditioning](https://arxiv.org/abs/2509.09677)
(NeurIPS 2025) shows per-step accuracy *degrades* as the context accumulates
the model's own errors. [Canonical-path deviation](https://arxiv.org/html/2602.19008)
measures it directly: one off-track tool call raises the probability the next
one is off-track by 22.7 points.

Both cannot describe the same process. The likely reconciliation is
**frailty** — pooling across tasks of differing difficulty makes hazard *look*
like it declines, because the survivors are disproportionately on easy tasks,
even while within-run hazard genuinely rises on each individual task.

Nobody has separated the two. Doing so requires many repeated runs of the
**same** task instance, and every existing study stops well short:

| study | repeats per task | on real tool calls? |
|---|---|---|
| Self-conditioning (2509.09677) | 100 | no — synthetic arithmetic |
| Beyond pass@1 (2603.29231) | 3 | yes |
| Canonical-path (2602.19008) | 3 | yes |
| τ²-bench | 4 | yes |
| METR | 8 | yes (x-axis is human-minutes) |
| BFCL | none documented | yes |

They stop there because on paid APIs, k≥30 across a swept chain length is a
serious bill. Locally, inference is free and the only cost is wall-clock —
which is what makes this measurable on a base-model MacBook Air.

## What this measures

Run the same tool-calling task many times each across a swept chain length,
with per-step outcome logging, and decompose the observed decay into task
heterogeneity versus genuine within-run dependence.

Planned analysis:

1. Weibull survival fit **with and without a per-task frailty term**. If the
   shape parameter moves toward 1 once frailty is included, the apparent
   declining hazard was heterogeneity all along.
2. **Beta-binomial dispersion** and the **ICC of per-step failures within
   run** — verified as absent from τ-bench, τ²-bench and BFCL, all three of
   which publish no confidence intervals or run-to-run variance at all.
3. Direct self-conditioning check: does per-step accuracy at step *i* depend
   on the number of prior errors in that run, holding task and step index fixed?

## Design rules

- **Raw facts, never derived verdicts.** There is no `success` field in the
  schema. Success is computed from `final_state` vs `goal_state` at analysis
  time, so the scorer can change without re-running inference.
- **The per-step vector is the point.** Hazard estimation needs what happened
  at every step, not just how the episode ended.
- **Paired seeds.** `seed = 1000 + repeat`, fixed per repeat and shared across
  models, so comparisons are paired.
- **Retry policy is recorded, not hidden.** "Succeeded with 3 retries" and
  "succeeded first try" are different results.

## The kill-test (run this first)

Can small local models emit a well-formed tool call at all? If a model cannot
clear a one-step call, it cannot participate — any decay curve measured with
it would be measuring the task, not the depth.

```bash
ollama serve
ollama pull qwen2.5:1.5b-instruct
ollama pull qwen2.5:0.5b-instruct

python scripts/killtest.py --models qwen2.5:1.5b-instruct qwen2.5:0.5b-instruct
```

Grading is four-level — `correct` / `bad_args` / `wrong_tool` /
`hallucinated_tool` / `no_call` — because "emitted a call" and "emitted the
right call" are different capabilities, and collapsing them is how a benchmark
produces a meaningless 100%.

## Layout

```
agentfrailty/
  schema.py      episode row + per-step records; append-only JSONL, fsync per row
  parsing.py     tool-call extraction, shared by the runner and the scorers
  sysmon.py      thermal/memory guards (reused from quantcost)
  runtimes/      ollama, llama.cpp, mlx adapters (reused from quantcost)
  envs/          deterministic tool environments  [not built yet]
  scorers/       goal-state grading              [not built yet]
scripts/         killtest, runner, analysis
tests/           parser and scorer tests
```

## Prior work by the same author

[quantcost](https://github.com/Ani-2003-HD/quantcost) — what quantization
actually costs in quality, measured on the same machine. The runtime adapters
and system monitors here are reused from it.

## License

TBD
