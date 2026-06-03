#!/usr/bin/env python3
"""8B thinking vs non-thinking 小样评测。

对 train 前 N 个任务、每个跑 K 次、两种模式(thinking / non-thinking),用本地
Qwen3-8B 当 agent、gpt-5 当 user、tau2 airline 当环境、官方 evaluator 算 reward。
agent 采样温度统一 0.8(两种模式相同,便于对比);top_p/top_k 用各模式 Qwen 推荐值。

模型只加载一次。每条结果实时写 outputs/eval_modes_results.jsonl,跑完打印汇总。

用法: python scripts/eval_modes.py [n_tasks=2] [k_runs=8]
"""
import os, sys, re, json, time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
for _k in ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_k, None)
from dotenv import load_dotenv
load_dotenv()
if os.environ.get("OPENAI_BASE_URL"):
    os.environ["OPENAI_API_BASE"] = os.environ["OPENAI_BASE_URL"]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tau2.data_model.message import (
    AssistantMessage, ToolCall, ToolMessage, MultiToolMessage, UserMessage, SystemMessage,
)
from tau2.agent.base_agent import HalfDuplexAgent
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner import build_environment, build_user, run_simulation
from tau2.evaluator.evaluator import EvaluationType
from tau2.data_model.tasks import RewardType
from tau2_airline_verl.data.splits import load_tasks
import tau2.evaluator.evaluator_nl_assertions as _nl
_nl.DEFAULT_LLM_NL_ASSERTIONS = os.environ.get("TAU2_NL_JUDGE_MODEL", "gpt-5")

MODEL_PATH = os.environ.get("QWEN3_8B_PATH", "./ckpts/Qwen3-8B")
USER_MODEL = os.environ.get("TAU2_USER_MODEL", "gpt-5")
TEMPERATURE = 0.8
OUT = "outputs/eval_modes_results.jsonl"


def _tool_schema(t):
    s = getattr(t, "openai_schema", None)
    if s is None and hasattr(t, "get_openai_schema"):
        s = t.get_openai_schema()
    return s


class _State:
    def __init__(self, system_messages, messages):
        self.system_messages = system_messages
        self.messages = messages


def _to_openai(msg):
    role = getattr(msg, "role", None)
    if isinstance(msg, SystemMessage) or role == "system":
        return [{"role": "system", "content": msg.content or ""}]
    if isinstance(msg, UserMessage) or role == "user":
        return [{"role": "user", "content": msg.content or ""}]
    if isinstance(msg, AssistantMessage) or role == "assistant":
        if getattr(msg, "tool_calls", None):
            return [{"role": "assistant", "content": msg.content or "",
                     "tool_calls": [{"type": "function",
                                     "function": {"name": tc.name, "arguments": tc.arguments}}
                                    for tc in msg.tool_calls]}]
        return [{"role": "assistant", "content": msg.content or ""}]
    if isinstance(msg, MultiToolMessage):
        return [{"role": "tool", "content": tm.content or ""} for tm in msg.tool_messages]
    if isinstance(msg, ToolMessage) or role == "tool":
        return [{"role": "tool", "content": msg.content or ""}]
    return [{"role": str(role), "content": getattr(msg, "content", "") or ""}]


class Qwen3LocalAgent(HalfDuplexAgent):
    def __init__(self, tools, domain_policy, model, tokenizer, template_kwargs=None,
                 max_new_tokens=2048, temperature=0.8, top_p=0.95, top_k=20):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.model = model
        self.tok = tokenizer
        self.tool_schemas = [_tool_schema(t) for t in tools]
        self.template_kwargs = template_kwargs or {}
        self.max_new_tokens = max_new_tokens
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        self.turn = 0

    def get_init_state(self, message_history=None):
        sys_p = ("You are a helpful airline customer service agent.\n\n"
                 f"## Domain Policy\n{self.domain_policy}\n\n"
                 "Follow the policy strictly. In each turn, either send a message to the "
                 "user OR make tool calls — never both, never empty.")
        return _State([SystemMessage(role="system", content=sys_p)],
                      list(message_history) if message_history else [])

    @torch.no_grad()
    def generate_next_message(self, message, state):
        self.turn += 1
        if message is not None:
            state.messages.append(message)
        oai = []
        for m in state.system_messages + state.messages:
            oai += _to_openai(m)
        prompt = self.tok.apply_chat_template(
            oai, tools=self.tool_schemas, add_generation_prompt=True, tokenize=False,
            **self.template_kwargs)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=True,
            temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
            pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        body = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        tcs = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", body, re.DOTALL)
        tool_calls = []
        for i, raw in enumerate(tcs):
            try:
                d = json.loads(raw)
                tool_calls.append(ToolCall(id=f"call_{self.turn}_{i}", name=d["name"],
                                           arguments=d.get("arguments", {}), requestor="assistant"))
            except Exception:
                pass
        if tool_calls:
            resp = AssistantMessage(role="assistant", content=None, tool_calls=tool_calls)
        else:
            content = re.sub(r"<tool_call>.*?</tool_call>", "", body, flags=re.DOTALL).strip()
            resp = AssistantMessage(role="assistant",
                                    content=content or "Could you clarify what you'd like me to help with?")
        state.messages.append(resp)
        return resp, state


# --- 加载模型(一次) ---
print(f"[load] {MODEL_PATH} on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()
assert "enable_thinking" in (tok.chat_template or ""), "该模型不是混合推理模型，无法切 thinking/non-thinking"
print("[load] done.", flush=True)

N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
K_RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
all_tasks = load_tasks("train")  # 本地分层 40/10 split(src/.../splits.py),非 submodule 旧 split
tasks = all_tasks[:N_TASKS]
MODES = [
    ("thinking",    {},                          dict(temperature=TEMPERATURE, top_p=0.95, top_k=20)),
    ("nonthinking", {"enable_thinking": False},   dict(temperature=TEMPERATURE, top_p=0.80, top_k=20)),
]
print(f"[plan] tasks={[t.id for t in tasks]} k_runs={K_RUNS} modes={[m[0] for m in MODES]} "
      f"temp={TEMPERATURE} -> total={len(tasks)*K_RUNS*len(MODES)} rollouts", flush=True)

open(OUT, "w").close()
results = []
t0 = time.time()
for mode_name, tkw, samp in MODES:
    for task in tasks:
        for run in range(K_RUNS):
            env = build_environment("airline")
            agent = Qwen3LocalAgent(env.get_tools(), env.get_policy(), model, tok,
                                    template_kwargs=tkw, **samp)
            user = build_user("user_simulator", env, task, llm=USER_MODEL)
            orch = Orchestrator(domain="airline", agent=agent, user=user, environment=env,
                                task=task, max_steps=30, max_errors=5, seed=run)
            try:
                res = run_simulation(orch, evaluation_type=EvaluationType.ALL)
                ri = res.reward_info
                bd = getattr(ri, "reward_breakdown", {}) or {}
                rec = dict(mode=mode_name, task=str(task.id), run=run,
                           reward=ri.reward,
                           db=bd.get(RewardType.DB), comm=bd.get(RewardType.COMMUNICATE),
                           nmsg=len(res.messages), term=str(res.termination_reason))
            except Exception as e:
                rec = dict(mode=mode_name, task=str(task.id), run=run, reward=None,
                           error=f"{type(e).__name__}: {str(e)[:200]}")
            results.append(rec)
            with open(OUT, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            el = time.time() - t0
            print(f"[{len(results):2d}/{len(tasks)*K_RUNS*len(MODES)}] {mode_name:11s} "
                  f"task={rec['task']} run={run} reward={rec.get('reward')} "
                  f"db={rec.get('db')} comm={rec.get('comm')} nmsg={rec.get('nmsg')} "
                  f"({el:.0f}s)", flush=True)

# --- 汇总 ---
print("\n" + "=" * 64 + "\n[SUMMARY]  (agent temperature=0.8)\n" + "=" * 64)
for mode_name, _, _ in MODES:
    print(f"\n## mode = {mode_name}")
    mode_rewards = []
    for task in tasks:
        rs = [r["reward"] for r in results
              if r["mode"] == mode_name and r["task"] == str(task.id) and r.get("reward") is not None]
        mode_rewards += rs
        if rs:
            mean = sum(rs) / len(rs)
            n1 = sum(1 for x in rs if x >= 0.999)
            print(f"  task {task.id}: rewards={rs}  mean={mean:.3f}  perfect={n1}/{len(rs)}")
        else:
            print(f"  task {task.id}: (no valid results)")
    if mode_rewards:
        print(f"  >>> mode mean reward = {sum(mode_rewards)/len(mode_rewards):.3f}  (n={len(mode_rewards)})")
print(f"\n[done] total {time.time()-t0:.0f}s. raw: {OUT}")
