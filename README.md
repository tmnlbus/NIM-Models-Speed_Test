# NVIDIA NIM — LLM Speed Benchmark

Dynamically discovers all models from the NVIDIA NIM API, pre-scans for
availability, then benchmarks every callable model on the same prompt. Results
are visualized as an animated bar chart.

## Setup

```bash
pip install openai python-dotenv
```

Add your API key to `.env`:
```
NVIDIA_API_KEY=your_key_here
```

## Run

```bash
# Fresh run — probes all ~185 models, benchmarks the ~104 callable ones
python benchmark.py --restart

# Quick mode — availability check only, no speed test
python benchmark.py --restart --quick

# Resume after interruption
python benchmark.py --resume

# Custom output file
python benchmark.py --restart --output my_results.json
```

Results are saved after every model — safe to interrupt and resume.

## How it works

1. **Discovery** — fetches the full model list from `GET /v1/models` (~185 models)
2. **Pre-scan** — quick `max_tokens=1` probe to check chat-completion availability
3. **Benchmark** — full streaming speed test on every callable model (tok/s, TTFT, latency)

Models that fail the pre-scan are recorded as `"unavailable"` and skipped.

## Visualize

```bash
python -m http.server 8080
# open http://localhost:8080/visualize.html
```

## Files

| File | Purpose |
|---|---|
| `benchmark.py` | Runs the benchmark |
| `models.json` | Static fallback list if API is unreachable |
| `results.json` | Output — generated after running |
| `visualize.html` | Animated bar chart |
