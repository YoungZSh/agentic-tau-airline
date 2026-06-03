#!/usr/bin/env python3
"""8B thinking vs non-thinking 评测 —— vllm 波次批处理版(并发)。

相比 eval_modes.py(HF+串行),这里:
  - agent 生成用 vllm,每个 group 的 8 条 rollout 同一波一起 batch(continuous batching);
  - gpt-5 用户回话用线程池 8 并发,隐藏 API 延迟;
  - 复用 tau2 的 Environment / UserSimulator / evaluate_simulation 保证 reward 口径正确。

波次 driver:对一个 group(同 mode+task 的 K 条),反复
  [批量 agent 生成(vllm)] -> [本地执行 tool / 线程池并发 gpt-5 user] -> ... 直到全部终止。

用法: python scripts/eval_modes_vllm.py [n_tasks=2] [k_runs=8]
"""
import os, sys, re, json, time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
for _k in ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_k, None)
from dotenv import load_dotenv
load_dotenv()
if os.environ.get("OPENAI_BASE_URL"):
    os.environ["OPENAI_API_BASE"] = os.environ["OPENAI_BASE_URL"]

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tau2.data_model.message import (
    AssistantMessage, ToolCall, ToolMessage, MultiToolMessage, UserMessage, SystemMessage,
)
from tau2.runner import build_environment, build_user
from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import RewardType
from tau2_airline_verl.data.splits import load_tasks
from tau2.user.user_simulator import UserSimulator
import tau2.evaluator.evaluator_nl_assertions as _nl
_nl.DEFAULT_LLM_NL_ASSERTIONS = os.environ.get("TAU2_NL_JUDGE_MODEL", "gpt-5")

MODEL_PATH = os.environ.get("QWEN3_8B_PATH", "./ckpts/Qwen3-8B")
USER_MODEL = os.environ.get("TAU2_USER_MODEL", "gpt-5")
TEMPERATURE = 0.8
MAX_AGENT_STEPS = 15           # 单条 agent 生成次数上限(防死循环)
FIRST_AGENT_MSG = "Hi! How can I help you today?"   # 复刻 tau2 Orchestrator 的开场白
OUT = "outputs/eval_modes_vllm_results.jsonl"
TS = "2026-06-02T00:00:00"     # SimulationRun 占位时间戳(evaluator 不关心)


def _tool_schema(t):
    s = getattr(t, "openai_schema", None)
    if s is None and hasattr(t, "get_openai_schema"):
        s = t.get_openai_schema()
    return s


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


def _parse(text, turn):
    """模型输出 -> AssistantMessage(纯 tool_calls 或 纯 content)。"""
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    tcs = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", body, re.DOTALL)
    tool_calls = []
    for i, raw in enumerate(tcs):
        try:
            d = json.loads(raw)
            tool_calls.append(ToolCall(id=f"call_{turn}_{i}", name=d["name"],
                                       arguments=d.get("arguments", {}), requestor="assistant"))
        except Exception:
            pass
    if tool_calls:
        return AssistantMessage(role="assistant", content=None, tool_calls=tool_calls)
    content = re.sub(r"<tool_call>.*?</tool_call>", "", body, flags=re.DOTALL).strip()
    return AssistantMessage(role="assistant",
                            content=content or "Could you clarify what you'd like me to help with?")


# --- 加载 vllm + tokenizer(一次) ---
print(f"[load] vllm {MODEL_PATH} on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
assert "enable_thinking" in (tok.chat_template or ""), "该模型不是混合推理模型，无法切 thinking/non-thinking"
llm = LLM(model=MODEL_PATH, dtype="bfloat16", gpu_memory_utilization=0.45,
          max_model_len=16384, enable_prefix_caching=True, trust_remote_code=True)
TOOLS = [_tool_schema(t) for t in build_environment("airline").get_tools()]
SYS_P = ("You are a helpful airline customer service agent.\n\n"
         f"## Domain Policy\n{build_environment('airline').get_policy()}\n\n"
         "Follow the policy strictly. In each turn, either send a message to the user "
         "OR make tool calls — never both, never empty.")
print("[load] done.", flush=True)


def _agent_generate_batch(convs, tkw, sp):
    """对一批 conv 各做一次 agent 生成(vllm 一次 batch)。原地把 AssistantMessage 追加进 conv['msgs']。"""
    prompt_token_ids = []
    for c in convs:
        oai = [{"role": "system", "content": SYS_P}]
        for m in c["msgs"]:
            oai += _to_openai(m)
        ids = tok.apply_chat_template(oai, tools=TOOLS, add_generation_prompt=True,
                                      tokenize=True, **tkw)
        prompt_token_ids.append(ids)
    outs = llm.generate([{"prompt_token_ids": ids} for ids in prompt_token_ids],
                        sp, use_tqdm=False)
    for c, o in zip(convs, outs):
        c["steps"] += 1
        am = _parse(o.outputs[0].text, c["steps"])
        c["msgs"].append(am)
        if am.tool_calls:
            for tc in am.tool_calls:
                try:
                    tm = c["env"].get_response(tc)
                except Exception as e:
                    tm = ToolMessage(id=tc.id, role="tool", requestor="assistant",
                                     content=f"Error: {e}")
                c["msgs"].append(tm)
            c["next"] = "agent"          # 工具结果回来后继续生成
        else:
            c["next"] = "user"           # 文本回复后等用户
        if c["steps"] >= MAX_AGENT_STEPS:
            c["done"], c["term"] = True, TerminationReason.MAX_STEPS


def run_group(mode_name, tkw, sp, task, k):
    """跑一个 group(k 条同 mode+task),波次推进。返回 k 条的结果记录。"""
    convs = []
    for j in range(k):
        env = build_environment("airline")
        user = build_user("user_simulator", env, task, llm=USER_MODEL)
        convs.append(dict(idx=j, env=env, user=user, ustate=user.get_init_state(),
                          msgs=[AssistantMessage(role="assistant", content=FIRST_AGENT_MSG)],
                          next="user", done=False, term=None, steps=0))

    def do_user(c):
        um, c["ustate"] = c["user"].generate_next_message(c["msgs"][-1], c["ustate"])
        return c, um

    while any(not c["done"] for c in convs):
        # 阶段A: 并发 gpt-5 用户回话
        ujobs = [c for c in convs if not c["done"] and c["next"] == "user"]
        if ujobs:
            with ThreadPoolExecutor(max_workers=len(ujobs)) as ex:
                for c, um in ex.map(do_user, ujobs):
                    c["msgs"].append(um)
                    if UserSimulator.is_stop(um):
                        c["done"], c["term"] = True, TerminationReason.USER_STOP
                    else:
                        c["next"] = "agent"
        # 阶段B: 批量 agent 生成(vllm)
        ajobs = [c for c in convs if not c["done"] and c["next"] == "agent"]
        if ajobs:
            _agent_generate_batch(ajobs, tkw, sp)

    # 评分
    out = []
    for c in convs:
        sim = SimulationRun(id=f"{mode_name}_{task.id}_{c['idx']}", task_id=str(task.id),
                            timestamp=TS, start_time=TS, end_time=TS, duration=0.0,
                            termination_reason=c["term"], messages=c["msgs"], mode="half_duplex")
        try:
            ri = evaluate_simulation(sim, task, EvaluationType.ALL, solo_mode=False, domain="airline")
            bd = getattr(ri, "reward_breakdown", {}) or {}
            rec = dict(mode=mode_name, task=str(task.id), run=c["idx"], reward=ri.reward,
                       db=bd.get(RewardType.DB), comm=bd.get(RewardType.COMMUNICATE),
                       nmsg=len(c["msgs"]), term=str(c["term"]))
        except Exception as e:
            rec = dict(mode=mode_name, task=str(task.id), run=c["idx"], reward=None,
                       error=f"{type(e).__name__}: {str(e)[:200]}", term=str(c["term"]))
        out.append(rec)
    return out


N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
K_RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
tasks = load_tasks("train")[:N_TASKS]  # 本地分层 40/10 split,非 submodule 旧 split
MODES = [
    ("thinking",    {},                         SamplingParams(temperature=TEMPERATURE, top_p=0.95, top_k=20, max_tokens=2048)),
    ("nonthinking", {"enable_thinking": False},  SamplingParams(temperature=TEMPERATURE, top_p=0.80, top_k=20, max_tokens=2048)),
]
print(f"[plan] tasks={[t.id for t in tasks]} k_runs={K_RUNS} modes={[m[0] for m in MODES]} "
      f"temp={TEMPERATURE} -> total={len(tasks)*K_RUNS*len(MODES)} rollouts (batch={K_RUNS})", flush=True)

open(OUT, "w").close()
results = []
t0 = time.time()
for mode_name, tkw, sp in MODES:
    for task in tasks:
        recs = run_group(mode_name, tkw, sp, task, K_RUNS)
        results += recs
        with open(OUT, "a") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
        rs = [r["reward"] for r in recs if r.get("reward") is not None]
        mean = sum(rs) / len(rs) if rs else float("nan")
        print(f"[group done] {mode_name:11s} task={task.id}  "
              f"rewards={[r.get('reward') for r in recs]}  mean={mean:.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

# --- 汇总 ---
print("\n" + "=" * 64 + "\n[SUMMARY]  vllm batch  (agent temperature=0.8)\n" + "=" * 64)
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
