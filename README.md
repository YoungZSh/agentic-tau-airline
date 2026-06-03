# agentic-tau-airline

τ³-bench **Airline** × **verl**: multi-turn tool-calling customer-service agent trained
with **LoRA + standard GRPO (RL-zero, no SFT)** on 2×A100 80GB. User simulator and
NL-assertion judge use external API (**gpt-5**); reward uses tau2's official
`reward_basis` (per-component subscores logged as curves).

Full plan: `docs/tau3_airline_qwen3_verl_grpo_plan.md`. The training/rollout/eval
pipeline (plan stages 1–3) is implemented; see Quickstart below.

## Layout

```
third_party/              verl + tau2-bench (read-only git submodules; not modified)
src/tau2_airline_verl/
  data/
    splits.py             train/held-out task splits (train=40, test=10; test stratified by scenario)
    airline_split.json    local 40/10 split ids + gpt-5 scenario labels (bypasses submodule split)
    build_parquet.py      airline tasks -> verl GRPO parquet (prompt=[system]; extra_info.task_id)
  env/                    plan §17 env/ — tau2 adapters (none modify tau2 source)
    airline_tool.py       airline Environment + tools + policy.md
    airline_interaction.py  tau2 UserSimulator wired to gpt-5
    reward.py             official reward + per-component subscores (gpt-5 NL judge)
  rollout/
    agent_loop.py         @register("tau2_airline") custom verl AgentLoop (Strategy B)
    tau2_session.py       synchronous tau2 env+user+evaluator wrapper for one trajectory
    conversion.py         tau2 <-> openai chat dict conversions + assistant builder
  agents/qwen3_prompt.py  system prompt = policy.md + native Qwen3 tool schema
  evaluation/
    run_tau2_eval.py      base rollout (GO/NO-GO) + held-out eval via tau2 runner + vLLM
    eval_passk.py         pass@1 / pass^k
    eval_tool_calls.py    tool-call accuracy / invalid-tool-call rate
    failure_analysis.py   §18.4 failure tagging
    report.py             Base vs GRPO comparison from trajectory JSONL
configs/                  grpo_qwen3_airline.yaml (reference) + tau2_agent_loop.yaml (registry)
scripts/
  setup/                  install submodules + download Qwen3-8B (hf-mirror)
  data/build_parquet.sh   build train/test parquet
  train/run_grpo.sh       GRPO + LoRA + multi-turn async (verl.trainer.main_ppo)
  train/run_grpo_smoke.sh tiny end-to-end smoke (2 tasks / n=2 / 1 step)
  eval/run_base_rollout.sh  stage 1 GO/NO-GO (base Qwen3-8B, train split)
  eval/run_eval.sh          stage 3 held-out eval (test split, ± LoRA adapter)
data/  outputs/           datasets + trajectories (tracked) ; experiment artifacts (gitignored)
tests/                    splits, reward, conversion, agent_loop (offline mask/reward checks)
```

## Quickstart (conda env `tau2verl`, a clone of `verl`)

```bash
# 1) submodules + install (once)
bash scripts/setup/install.sh
cp .env.example .env && $EDITOR .env       # set OPENAI_API_KEY (+ OPENAI_BASE_URL), QWEN3_8B_PATH

# 2) model weights (~16GB, via hf-mirror, no proxy)
bash scripts/setup/download_qwen3_8b.sh

# 3) offline tests (no GPU/API)
conda run -n tau2verl python -m pytest tests/ -q

# 4) build verl datasets (train=40 / test=10)
bash scripts/data/build_parquet.sh

# 5) stage 1 — base rollout GO/NO-GO (starts a local vLLM server, runs tau2 eval)
bash scripts/eval/run_base_rollout.sh 8            # -> notes/stage1_base_rollout.md
#    smoke first: bash scripts/eval/run_base_rollout.sh 2 --task-ids 0 1

# 6) stage 2 — GRPO training (2×A100)
bash scripts/train/run_grpo_smoke.sh               # tiny end-to-end smoke
bash scripts/train/run_grpo.sh                     # full run

# 7) stage 3 — held-out eval + Base vs GRPO report
LORA_PATH=<trained_adapter_dir> bash scripts/eval/run_eval.sh 8
conda run -n tau2verl python -m tau2_airline_verl.evaluation.report --base base.jsonl --grpo grpo.jsonl
```

## How it fits together

`build_parquet` emits one row per task carrying only the system prompt (airline
`policy.md`); the multi-turn conversation is generated at rollout time inside
`Tau2AirlineAgentLoop` (`agent_loop.py`): the policy turn goes through verl's LLM
server, tool calls hit tau2's `Environment` (real DB mutations), natural-language
turns go to tau2's gpt-5 `UserSimulator`, and the trajectory is scored by tau2's
official `evaluate_simulation` (DB hash + communicate). The scalar reward rides
back to verl via `AgentLoopOutput.reward_score` (→ `rm_scores[-1]`), so the
default `naive` reward manager needs no custom reward function. Reference =
adapter-off base model (LoRA), assistant tokens get loss (`response_mask=1`),
tool/user tokens are masked out.
