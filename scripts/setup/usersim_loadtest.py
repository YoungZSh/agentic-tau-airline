"""Concurrency load test for the local user-sim vllm server (port 8011).

Two scenarios, sized from the REAL rollout user-sim length distribution
(mean 76 / p95 163 / max 255 tok, thinking off):
  - "realistic": max_tokens=256, natural stop (~matches the real distribution)
  - "worst":     min_tokens=230 + ignore_eos, forces ~max-length decode (the
                 p99/tail case) to bound latency.
Prompt padded to ~1.2k tok to mimic the user-sim's real context. Run in `vllm` env.
"""
import asyncio, time, statistics
import aiohttp

URL = "http://127.0.0.1:8011/v1/chat/completions"
MODEL = "qwen3.6-usersim"

SYS = ("You are a customer contacting airline support. Scenario: you booked a "
       "round-trip flight and now need to review several upcoming reservations, "
       "cancel any with a segment longer than 4 hours, and upgrade short trips to "
       "business; you have your booking references and want fees itemized before "
       "any change. Stay strictly in character as the customer, reply naturally, "
       "and never reveal you are an AI. ") * 16   # ~1.2k tokens
MESSAGES = [
    {"role": "system", "content": SYS},
    {"role": "user", "content": "Hello, this is airline support. How can I help you today?"},
]


def payload(mode):
    p = {"model": MODEL, "messages": MESSAGES, "temperature": 0.7,
         "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}}
    if mode == "worst":
        p["min_tokens"] = 230
        p["ignore_eos"] = True
    return p


async def one(session, mode):
    t0 = time.time()
    async with session.post(URL, json=payload(mode)) as r:
        d = await r.json()
    return time.time() - t0, d["usage"]["completion_tokens"]


async def run_level(session, conc, mode):
    t0 = time.time()
    res = await asyncio.gather(*[one(session, mode) for _ in range(conc)])
    wall = time.time() - t0
    lats = sorted(r[0] for r in res)
    toks = sum(r[1] for r in res)
    p = lambda q: lats[min(len(lats) - 1, int(q * len(lats)))]
    print(f"  [{mode:8s}] conc={conc:3d} | wall={wall:6.2f}s | {conc/wall:5.1f} req/s | "
          f"{toks/wall:7.1f} out-tok/s | avg_out={toks/conc:5.0f} | "
          f"lat p50={p(.5):5.2f} p95={p(.95):5.2f} max={lats[-1]:5.2f}s")


async def main():
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout,
                                     connector=aiohttp.TCPConnector(limit=0)) as s:
        print("[warmup x3]")
        await asyncio.gather(*[one(s, "worst") for _ in range(3)])  # warm long-decode kernels
        for mode in ("realistic", "worst"):
            for conc in (32, 64):
                await run_level(s, conc, mode)


asyncio.run(main())
