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

> A small project that compares two personal chat assistant deployments.

- OSS assistant: https://huggingface.co/spaces/dart599/luna-oss
- Foundation assistant: https://luna-foundational-assistant.streamlit.app/

The OSS assistant runs a Streamlit UI with `Qwen/Qwen2.5-1.5B-Instruct`.
The foundation assistant uses Gemini through a separate Streamlit app in
`foundation_assistant/`.

---

## Setup Instructions

### OSS Assistant

```bash
pip install -r requirements.txt
streamlit run app.py
```

The OSS model downloads from Hugging Face on first run, so startup can be slow
and requires several GB of free disk space.

### Foundation Assistant

```bash
cd foundation_assistant
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

Add your Gemini key to `foundation_assistant/.env`:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

## Architecture Decisions

- The OSS assistant uses a local Hugging Face transformer model to demonstrate
  open-source deployment and model ownership.
- The foundation assistant uses Gemini to reduce latency and avoid local model
  hosting requirements.
- Streamlit was used for both applications to keep the UI simple and consistent.
- Short-term memory is implemented by sending recent conversation turns with
  each request.
- Evaluation is handled with custom factual, bias-sensitive, and adversarial
  prompts, with results saved as Markdown/JSONL.
- Observability for the foundation assistant is stored as local JSONL trace logs.

## Tradeoffs Made

- Free CPU deployment for the OSS model avoids GPU cost but causes high latency.
- Gemini improves response speed and safety behavior but depends on an external
  API and rate limits.
- Streamlit is fast to build and deploy, but less flexible than a custom
  React/Next.js frontend.
- The OSS model gives more deployment control, but local evaluation needs disk
  space and compute.
- Guardrails are lightweight and local; they catch obvious unsafe prompts but
  are not a full safety moderation system.

## Evaluation

The assistants were evaluated on:

- factual prompts for hallucination rate
- sensitive prompts for bias and harmful stereotypes
- adversarial prompts for jailbreak resistance and refusal handling

Reports are stored in:

```text
foundation_assistant/eval_results/
```

## Improvements With More Time

- Add a proper LLM-as-judge evaluator for more nuanced scoring.
- Run OSS evaluation on a GPU machine or cloud VM to avoid local disk and CPU
  limits.
- Add a stronger moderation layer before model calls.
- Store conversations and traces in a database instead of local JSONL files.
- Add authentication and per-user chat history.
- Build a custom Next.js frontend if deploying the foundation assistant on
  Vercel.
- Add automated CI checks for eval regressions.

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
