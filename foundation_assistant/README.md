# Foundation Model Assistant

A Streamlit personal assistant using Gemini foundation models instead of local
OSS inference. It supports:

- Gemini API
- Multi-turn conversation
- Short-term conversational memory
- Basic assistant behavior through a system prompt
- Local guardrails for unsafe requests
- JSONL observability logs
- A small eval suite for helpfulness, memory, guardrail behavior, and latency

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add `GEMINI_API_KEY` to `.env`.

## Run the App

```bash
streamlit run app.py
```

## Run Evals

```bash
python evals.py
```

Eval results are written to `eval_results/results.jsonl`.

## Compare OSS and Gemini Assistants

Run the hallucination, bias/harm, and content-safety comparison:

```bash
python compare_assistants.py
```

This writes:

```text
eval_results/comparison_report.md
eval_results/assistant_comparison.jsonl
```

Use `--skip-oss` or `--skip-gemini` if you only want to run one assistant.

## Observability

Every model call writes a trace event to:

```text
logs/events.jsonl
```

Each event includes provider, model, latency, trace ID, guardrail status, and
message count. Sensitive token-like strings are redacted before logging.

## Notes

Gemini is used through the Google Gen AI SDK. The app keeps short-term memory
by sending recent conversation turns with each request.
