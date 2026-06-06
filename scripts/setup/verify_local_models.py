"""Verify user-sim + judge wiring against the local vllm server.

Two roles, two thinking modes (decoupled, see airline_interaction.py / reward.py):
  - user-sim: thinking ON. With the server run via `--reasoning-parser qwen3`, the
    <think> block must land in `reasoning_content` and the OpenAI `content` (what
    tau2 reads as the user reply) must stay CLEAN. If `content` still contains
    <think>, the reasoning parser is NOT active -> restart serve_usersim.sh.
  - judge: thinking OFF, content clean.

Run in tau2verl env after .env is present:
  conda run -n tau2verl python scripts/setup/verify_local_models.py
"""
import os

from dotenv import load_dotenv

load_dotenv(".env")
if os.environ.get("OPENAI_BASE_URL"):
    os.environ["OPENAI_API_BASE"] = os.environ["OPENAI_BASE_URL"]

from openai import OpenAI  # noqa: E402

from tau2.data_model.message import SystemMessage, UserMessage  # noqa: E402
from tau2.utils.llm_utils import generate  # noqa: E402
from tau2_airline_verl.env.airline_interaction import _user_llm_args_from_env  # noqa: E402
from tau2_airline_verl.env.reward import set_nl_judge_model  # noqa: E402
import tau2.evaluator.evaluator_nl_assertions as nl  # noqa: E402


def has_think(s):
    # The opening <think> is pre-filled into the prompt by the chat template, so a
    # thinking turn's OUTPUT never contains <think> — only the closing </think>
    # (followed by the real answer). Without --reasoning-parser the whole
    # reasoning + </think> + answer lands in `content`; the closing tag is the leak
    # signal. With the parser, reasoning -> reasoning_content and content is clean.
    s = s or ""
    return "</think>" in s or "<think>" in s


USER_MODEL = os.environ["TAU2_USER_MODEL"]
ok = True

print("=== user-sim path (thinking ON; tau2 generate() call path) ===")
user_args = _user_llm_args_from_env(USER_MODEL)
print("  llm_args =", user_args)
m = generate(
    model=USER_MODEL,
    messages=[
        SystemMessage(role="system", content="You are a customer contacting airline "
                      "support. You want to cancel your flight. Reply briefly, in character."),
        UserMessage(role="user", content="Hello, this is airline support. How can I help you today?"),
    ],
    num_retries=0,
    **user_args,
)
print("  content =", repr((m.content or "")[:200]))
if has_think(m.content):
    ok = False
    print("  !! <think> LEAKED into content -> reasoning parser NOT active.")
    print("     Restart the server: bash scripts/setup/serve_usersim.sh (with --reasoning-parser qwen3)")
else:
    print("  OK: content is clean (no <think> leak)")

# Raw probe: confirm the server actually separated reasoning into reasoning_content.
print("=== raw probe (OpenAI client) -> reasoning_content separation ===")
client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ.get("OPENAI_API_KEY", "x"))
served = USER_MODEL.split("/", 1)[-1]  # strip the litellm "openai/" routing prefix
resp = client.chat.completions.create(
    model=served,
    messages=[
        {"role": "system", "content": "You are a customer. You want to cancel your flight. Reply in character."},
        {"role": "user", "content": "Hello, this is airline support. How can I help you today?"},
    ],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
msg = resp.choices[0].message
reasoning = getattr(msg, "reasoning_content", None)
print("  reasoning_content present?", bool(reasoning), "(len=%d)" % len(reasoning or ""))
print("  content =", repr((msg.content or "")[:160]))
if not reasoning:
    print("  !! reasoning_content empty -> parser inactive or model didn't think this turn")

print("=== judge path (thinking OFF) ===")
set_nl_judge_model()
m2 = generate(
    model=nl.DEFAULT_LLM_NL_ASSERTIONS,
    messages=[
        SystemMessage(role="system", content="You are an evaluator. Answer with one word."),
        UserMessage(role="user", content="Did the agent help? Reply exactly: yes"),
    ],
    num_retries=0,
    **nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS,
)
print("  content =", repr((m2.content or "")[:120]))
if has_think(m2.content):
    ok = False
    print("  !! judge content has <think> (judge should stay thinking-off)")
else:
    print("  OK: judge content clean")

print("\n" + ("ALL OK: user-sim thinks with clean content; judge clean." if ok
             else "PROBLEM: see !! lines above (likely need to restart serve_usersim.sh)."))
