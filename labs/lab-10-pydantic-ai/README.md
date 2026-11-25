# Lab 10: Type-Safe Agents with PydanticAI

Production pattern for **schema-validated agent outputs**, **dependency injection (`RunContext`)**, and **EU AI Act Article 14 human-oversight gating** using [PydanticAI](https://ai.pydantic.dev/).

## Run Locally

```bash
python labs/lab-10-pydantic-ai/agent.py
```

- Runs in **deterministic offline fallback** mode by default when no API key is set.
- Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` (with `pydantic-ai` installed) to run against a live model.
