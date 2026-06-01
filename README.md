# agentic-tau-airline

τ³-bench **Airline** × **verl**: multi-turn tool-calling customer-service agent trained
with **LoRA + standard GRPO (RL-zero, no SFT)** on 2×A100 80GB. User simulator and
NL-assertion judge use external API (**gpt-5**); reward uses tau2's official
`reward_basis` (per-component subscores logged as curves).

Full plan: `docs/tau3_airline_qwen3_verl_grpo_plan.md`. This repo currently implements
**stage 0** (environment integration + understanding); training is stage 2.

## Layout

```
src/tau2_airline_verl/
  data/splits.py        train/held-out task splits (train=30, test=20)
  tau2env/factory.py    airline Environment + tools (wraps tau2)
  usersim/factory.py    tau2 UserSimulator wired to gpt-5
  reward/evaluate.py    official reward + per-component subscores (gpt-5 NL judge)
  rollout/
    tau2_agent_loop.py  @register("tau2_airline") custom verl AgentLoop (Strategy B; skeleton)
    conversion.py       tau2 <-> openai chat dict conversions
scripts/
  install_tau2.sh                clone + pip install tau2 into tau2verl env
  download_qwen3_8b.sh           fetch Qwen/Qwen3-8B
  understand_airline.py          stage-0 understanding notes -> notes/
  sanity_rollout_orchestrator.py stage-0 main acceptance (end-to-end, external API)
configs/                env / reward / usersim / verl configs
tests/                  splits, reward adapter, conversion
```

## Quickstart (in conda env `tau2verl`, a clone of `verl`)

```bash
# 0) clone env (protects the pristine verl env) — once
conda create --clone verl -n tau2verl -y

# 1) install tau2 + this package
bash scripts/install_tau2.sh

# 2) secrets
cp .env.example .env && $EDITOR .env     # set OPENAI_API_KEY

# 3) tests + understanding notes (no API)
/home/yzs/miniconda3/envs/tau2verl/bin/python -m pytest tests/ -q
/home/yzs/miniconda3/envs/tau2verl/bin/python scripts/understand_airline.py

# 4) end-to-end sanity (uses gpt-5)  <-- stage-0 acceptance
/home/yzs/miniconda3/envs/tau2verl/bin/python scripts/sanity_rollout_orchestrator.py --task 0

# 5) AgentLoop registers
/home/yzs/miniconda3/envs/tau2verl/bin/python -c "import tau2_airline_verl.rollout.tau2_agent_loop; print('registered')"
```
