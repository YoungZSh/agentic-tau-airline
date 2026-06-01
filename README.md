# agentic-tau-airline

τ³-bench **Airline** × **verl**: multi-turn tool-calling customer-service agent trained
with **LoRA + standard GRPO (RL-zero, no SFT)** on 2×A100 80GB. User simulator and
NL-assertion judge use external API (**gpt-5**); reward uses tau2's official
`reward_basis` (per-component subscores logged as curves).

Full plan: `docs/tau3_airline_qwen3_verl_grpo_plan.md`. This repo currently implements
**stage 0** (environment integration + understanding); training is stage 2.

## Layout

```
third_party/              verl + tau2-bench (read-only git submodules; not modified)
src/tau2_airline_verl/
  data/splits.py            train/held-out task splits (train=30, test=20)
  env/                      plan §17 env/ — tau2 adapters (none modify tau2 source)
    airline_tool.py         airline Environment + tools + policy.md
    airline_interaction.py  tau2 UserSimulator wired to gpt-5
    reward.py               official reward + per-component subscores (gpt-5 NL judge)
  rollout/
    agent_loop.py           @register("tau2_airline") custom verl AgentLoop (Strategy B; skeleton)
    conversion.py           tau2 <-> openai chat dict conversions
  agents/qwen3_prompt.py    system prompt = policy.md + native Qwen3 tool schema
  evaluation/               pass@1/pass^k, tool-call acc, failure analysis (stage 3)
  utils/paths.py            centralized path resolution from .env (no hard-coded paths)
configs/                    grpo / rollout / paths / reward / usersim / tau2_agent_loop
scripts/
  install.sh                init submodules + install verl/tau2/this pkg + smoke test
  setup/download_qwen3_8b.sh  fetch Qwen/Qwen3-8B
data/  outputs/             datasets + trajectories (tracked) ; experiment artifacts (gitignored)
tests/                      splits, reward adapter, conversion
```

## Quickstart (in conda env `tau2verl`, a clone of `verl`)

```bash
# 0) clone env (protects the pristine verl env) — once
conda create --clone verl -n tau2verl -y

# 1) init submodules (verl + tau2-bench) + install them + this package
bash scripts/setup/install.sh

# 2) secrets
cp .env.example .env && $EDITOR .env     # set OPENAI_API_KEY

# 3) tests (no API)
conda run -n tau2verl python -m pytest tests/ -q

# 4) AgentLoop registers
conda run -n tau2verl python -c "import tau2_airline_verl.rollout.agent_loop; print('registered')"
```
