# NVIDIA NIM — LLM Speed Benchmark

Benchmarks 20 free NVIDIA NIM models on the same prompt and visualizes results as an animated bar chart.

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
# Fresh run
python benchmark.py --restart

# Resume after interruption
python benchmark.py --resume
```

Results are saved to `results.json` after every model — safe to interrupt and resume.

## Visualize

Serve the folder and open the chart:
```bash
python -m http.server 8080
# open http://localhost:8080/visualize.html
```

## Files

| File | Purpose |
|---|---|
| `benchmark.py` | Runs the benchmark |
| `models.json` | List of models to test |
| `results.json` | Output — generated after running |
| `visualize.html` | Animated bar chart |
