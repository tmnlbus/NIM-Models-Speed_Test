import os
import json
import time
import argparse
import urllib.request
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description="NVIDIA NIM LLM Speed Benchmark")
group = parser.add_mutually_exclusive_group()
group.add_argument("--resume", action="store_true", help="Continue from last completed model (default)")
group.add_argument("--restart", action="store_true", help="Delete results file and start from scratch")
parser.add_argument("--quick", action="store_true", help="Light probe only — availability check, no speed test")
parser.add_argument("--output", default="results.json", help="Output results file (default: results.json)")
args = parser.parse_args()

API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = "https://integrate.api.nvidia.com/v1"
PROMPT = "What is the capital of India? Please answer in one sentence."
RESULTS_FILE = args.output
RATE_LIMIT_WAIT = 12
REQUEST_TIMEOUT = 60

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=REQUEST_TIMEOUT)


# ── helpers ─────────────────────────────────────────────────────────────────


def make_label(model_id):
    parts = model_id.split("/")
    name_part = parts[-1] if len(parts) >= 2 else parts[0]
    return name_part.replace("-", " ").replace("_", " ").title()


def fetch_model_list():
    """GET /v1/models and return a sorted list of model dicts."""
    print("Fetching model list from NVIDIA NIM API...")
    req = urllib.request.Request(f"{BASE_URL}/models")
    req.add_header("Authorization", f"Bearer {API_KEY}")
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  API fetch failed ({e}). Falling back to models.json.")
        with open("models.json") as f:
            return json.load(f)

    raw = data.get("data", [])
    raw.sort(key=lambda m: m["id"])
    result = [{"id": m["id"], "label": make_label(m["id"]), "owned_by": m.get("owned_by", "")} for m in raw]
    print(f"  Found {len(result)} models in catalog.")
    return result


def quick_probe(model_id):
    """Lightweight check whether a model accepts chat completions."""
    client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1,
    )
    return True


def benchmark_model(model_id, label):
    """Full streaming speed benchmark on a single model."""
    print(f"\n  ▶ {label} ({model_id})")
    try:
        start = time.perf_counter()
        token_count = 0
        first_token_time = None

        stream = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.7,
            max_tokens=300,
            stream=True,
        )

        full_text = ""
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and getattr(delta, "content", None):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                full_text += delta.content
                token_count += 1
                print(".", end="", flush=True)

        elapsed = time.perf_counter() - start
        ttft = (first_token_time - start) if first_token_time else None
        tps = token_count / elapsed if elapsed > 0 else 0

        print(f"\n     {token_count} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")
        return {
            "id": model_id,
            "label": label,
            "status": "ok",
            "total_time_s": round(elapsed, 3),
            "tokens": token_count,
            "tokens_per_second": round(tps, 2),
            "ttft_s": round(ttft, 3) if ttft else None,
            "response_preview": full_text[:120],
        }

    except Exception as e:
        msg = str(e)
        print(f"\n     ERROR: {msg[:120]}")
        return {
            "id": model_id,
            "label": label,
            "status": "error",
            "error": msg[:200],
            "total_time_s": None,
            "tokens": 0,
            "tokens_per_second": 0,
            "ttft_s": None,
        }


# ── main ─────────────────────────────────────────────────────────────────────

if args.restart and os.path.exists(RESULTS_FILE):
    os.remove(RESULTS_FILE)
    print("Deleted results.json — starting fresh.")

try:
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    done_ids = {r["id"] for r in results}
    print(f"Loaded {len(results)} existing results ({len(done_ids)} unique models).")
except FileNotFoundError:
    results = []
    done_ids = set()

models = fetch_model_list()

if not done_ids:
    print(f"Starting fresh — {len(models)} models to process.\n")
else:
    remaining = sum(1 for m in models if m["id"] not in done_ids)
    print(f"Resuming — {len(done_ids)} already done, {remaining} remaining.\n")

for model in models:
    mid = model["id"]
    if mid in done_ids:
        print(f"  skip (done): {model['label']}")
        continue

    # ── Step 1: quick probe ──
    print(f"  · {model['label']} ({mid}) — probing ...", end=" ")
    try:
        is_callable = quick_probe(mid)
        print("OK")
    except RateLimitError:
        print("RATE LIMITED")
        time.sleep(RATE_LIMIT_WAIT * 2)
        continue
    except Exception as e:
        print(f"FAIL ({str(e)[:60]})")
        is_callable = False

    time.sleep(RATE_LIMIT_WAIT)

    if not is_callable:
        result = {
            "id": mid,
            "label": model["label"],
            "status": "unavailable",
            "error": "not callable via chat completions",
        }
        results.append(result)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  ✗ {model['label']}: unavailable (skipping speed test)")
        continue

    if args.quick:
        result = {
            "id": mid,
            "label": model["label"],
            "status": "ok",
            "note": "quick probe only",
        }
        results.append(result)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  ✓ {model['label']}: available (quick mode)")
        continue

    # ── Step 2: full speed benchmark ──
    result = benchmark_model(mid, model["label"])
    results.append(result)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    time.sleep(RATE_LIMIT_WAIT)

# ── summary ─────────────────────────────────────────────────────────────────

print(f"\nDone! Results saved to {RESULTS_FILE}")

ok_results = [r for r in results if r.get("status") == "ok" and r.get("tokens_per_second")]
if ok_results:
    print("\nTop 5 by tokens/second:")
    ranked = sorted(ok_results, key=lambda x: x["tokens_per_second"], reverse=True)
    for i, r in enumerate(ranked[:5], 1):
        print(f"  {i}. {r['label']}: {r['tokens_per_second']} tok/s")

total = len(results)
ok_count = sum(1 for r in results if r.get("status") == "ok")
unavail = sum(1 for r in results if r.get("status") == "unavailable")
errors = sum(1 for r in results if r.get("status") == "error")
print(f"\nSummary: {ok_count} ok, {unavail} unavailable, {errors} errors out of {total} total models")
