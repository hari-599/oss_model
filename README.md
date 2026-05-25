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
