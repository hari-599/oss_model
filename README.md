---
title: Luna OSS Chat Assistant
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Luna OSS Chat Assistant

A public Hugging Face Space running a simple Streamlit chat UI with
`Qwen/Qwen2.5-1.5B-Instruct`.

## Notes

- The first request may be slow because the model has to load.
- Free CPU Spaces can be very slow for a 1.5B parameter model.
- Do not commit `.env` or Hugging Face access tokens.

## Cost and Latency Benchmark

Run this after installing dependencies:

```bash
python benchmark.py
```

The script loads the same OSS model used by the app, sends a few test prompts,
and prints a Markdown table with latency, tokens per second, and estimated cost
per 1,000 chats for common Hugging Face Space hardware options.
