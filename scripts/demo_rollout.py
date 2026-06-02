#!/usr/bin/env python3
"""演示：用本地 Qwen3-8B 作为 policy，在 tau2 airline 的一个 train 任务上跑一条完整 rollout。

  - policy: 本地 Qwen3-8B (transformers, GPU, thinking 模式, Hermes 原生 tool-calling)
  - user  : gpt-5 (litellm -> OPENAI_BASE_URL, 启动时已清掉本机代理以直连)
  - env   : tau2 airline Environment (本地, 每条轨迹独立 DB)
  - reward: tau2 官方 evaluator (EvaluationType.ALL), NL 断言 judge 也用 gpt-5

仅用于人工查看一条轨迹长什么样，不是训练。用法: python scripts/demo_rollout.py [task_id]
"""
import os, sys, re, json

# --- 0. 环境：锁定一张空闲 GPU；让 litellm 绕过本机代理直连 yunwu.ai ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
for _k in ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_k, None)
from dotenv import load_dotenv
load_dotenv()
if os.environ.get("OPENAI_BASE_URL"):
    os.environ["OPENAI_API_BASE"] = os.environ["OPENAI_BASE_URL"]  # litellm 兼容两种变量名

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tau2.data_model.message import (
    AssistantMessage, ToolCall, ToolMessage, MultiToolMessage,
    UserMessage, SystemMessage,
)
from tau2.agent.base_agent import HalfDuplexAgent
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner import build_environment, build_user, run_simulation
from tau2.evaluator.evaluator import EvaluationType
from tau2.domains.airline.environment import get_tasks
import tau2.evaluator.evaluator_nl_assertions as _nl
_nl.DEFAULT_LLM_NL_ASSERTIONS = os.environ.get("TAU2_NL_JUDGE_MODEL", "gpt-5")

MODEL_PATH = os.environ.get("DEMO_MODEL_PATH") or os.environ.get("QWEN3_8B_PATH", "./ckpts/Qwen3-8B")
USER_MODEL = os.environ.get("TAU2_USER_MODEL", "gpt-5")


# --- 1. 本地 Qwen3 agent ---
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
    """tau2 Message -> OpenAI chat dict(s)。arguments 保持 dict（Qwen3 模板要求）。"""
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
    def __init__(self, tools, domain_policy, model, tokenizer, max_new_tokens=2048,
                 temperature=0.6, top_p=0.95, top_k=20):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.model = model
        self.tok = tokenizer
        self.tool_schemas = [_tool_schema(t) for t in tools]
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.turn = 0

    def get_init_state(self, message_history=None):
        sys_p = (
            "You are a helpful airline customer service agent.\n\n"
            f"## Domain Policy\n{self.domain_policy}\n\n"
            "Follow the policy strictly. In each turn, either send a message to the "
            "user OR make tool calls — never both, never empty."
        )
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
            oai, tools=self.tool_schemas, add_generation_prompt=True, tokenize=False)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=True,
            temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
            pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        think = "".join(re.findall(r"<think>(.*?)</think>", text, re.DOTALL)).strip()
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
            self._log(think, None, tool_calls)
        else:
            content = re.sub(r"<tool_call>.*?</tool_call>", "", body, flags=re.DOTALL).strip()
            if not content:
                content = "Could you clarify what you'd like me to help with?"
            resp = AssistantMessage(role="assistant", content=content)
            self._log(think, content, None)

        state.messages.append(resp)
        return resp, state

    def _log(self, think, content, tool_calls):
        print(f"\n[AGENT turn {self.turn}]")
        if think:
            print(f"  <think> {think[:300]}{'…' if len(think) > 300 else ''}")
        if tool_calls:
            for tc in tool_calls:
                print(f"  -> TOOL_CALL {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})")
        if content:
            print(f"  -> SAY: {content}")
        sys.stdout.flush()


# --- 2. 加载本地模型 ---
print(f"[load] {MODEL_PATH}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}) ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()
# 混合推理模型(8B)模板里有 enable_thinking 开关；纯非思考模型(4B-Instruct-2507)没有。
# 据此选 Qwen 官方推荐采样：thinking=0.6/0.95，non-thinking=0.7/0.8。
_is_thinking = "enable_thinking" in (tok.chat_template or "")
_samp = dict(temperature=0.6, top_p=0.95, top_k=20) if _is_thinking \
    else dict(temperature=0.7, top_p=0.8, top_k=20)
print(f"[load] done. mode={'thinking' if _is_thinking else 'non-thinking'} sampling={_samp}", flush=True)

# --- 3. 任务 + 三方组件 ---
TASK_ID = sys.argv[1] if len(sys.argv) > 1 else "0"
tasks = get_tasks("train")
task = next((t for t in tasks if str(t.id) == TASK_ID), tasks[0])
ins = task.user_scenario.instructions
print(f"\n[task] id={task.id}", flush=True)
print(f"  purpose: {task.description.purpose if task.description else ''}")
print(f"  reason_for_call: {getattr(ins, 'reason_for_call', ins)}")
print(f"  known_info: {getattr(ins, 'known_info', '')}")
print(f"  reward_basis: {task.evaluation_criteria.reward_basis}")

env = build_environment("airline")
agent = Qwen3LocalAgent(tools=env.get_tools(), domain_policy=env.get_policy(),
                        model=model, tokenizer=tok, **_samp)
user = build_user("user_simulator", env, task, llm=USER_MODEL)
orch = Orchestrator(domain="airline", agent=agent, user=user, environment=env,
                    task=task, max_steps=30, max_errors=5, seed=42)

print("\n" + "=" * 72 + "\n[rollout] start (Qwen3 agent  x  gpt-5 user)\n" + "=" * 72, flush=True)
result = run_simulation(orch, evaluation_type=EvaluationType.ALL)

# --- 4. 完整轨迹 + reward ---
print("\n" + "=" * 72 + "\n[full trajectory]\n" + "=" * 72)
for m in result.messages:
    role = m.role.value if hasattr(m.role, "value") else m.role
    if getattr(m, "tool_calls", None):
        for tc in m.tool_calls:
            print(f"[{role}] TOOL_CALL {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})")
    elif role == "tool":
        print(f"[tool] {(m.content or '')[:400]}")
    else:
        print(f"[{role}] {m.content or ''}")

print("\n" + "=" * 72 + "\n[result]\n" + "=" * 72)
ri = result.reward_info
print(f"termination_reason: {result.termination_reason}")
print(f"num_messages: {len(result.messages)}")
if ri is not None:
    print(f"REWARD: {ri.reward}")
    print(f"reward_breakdown: {getattr(ri, 'reward_breakdown', None)}")
    print(f"db_check: {getattr(ri, 'db_check', None)}")
else:
    print("reward_info: None")
