# Experiment configuration

The manuscript's recorded run used the versioned trio of trader-persona prompts
and a fixed deck of binary market scenarios. These are benchmark prompts and
scenario labels. The repository's analysis evaluates vote records.

| Setting | Location |
|---|---|
| Persona prompts, versions, authored scenarios and truth labels | [`ntqr.llm.personas`](../python/src/ntqr/llm/personas.py) |
| Default model roster, backend, temperature, timeout, and refresh settings | [Multi-model runner](../scripts/llm/multi_llm_personas_evaluation.py) and its imported model defaults |
| Saved settings, parse outcomes, and per-model results | [Recorded sweep](../outputs/llm/multi_llm_personas_evaluation/README.md) |
| General CLI, JSON configuration and labeled-dataset input | [Orchestrator configuration](../scripts/_config.py) and [config examples](../scripts/configs/README.md) |
| Title, authors, bibliography, formats, and PDF styling | [Canonical manuscript directory](manuscript/README.md) |

## Live collection

The multi-model runner discovers available models before it reads the prompt
cache. Starting it with the default `refresh=false` can still call Ollama to
fill cache misses. A live run creates new observations and may replace
analysis outputs, so use a separate checkout when collecting new data.
Use the [offline verification procedure](VERIFICATION.md) to check the
recorded study with no model server.

For an explicitly planned live collection, install the requested Ollama model
tags and start the local service, then run from that separate checkout:

```bash
uv sync --locked --extra dev
uv run --no-sync python scripts/llm/multi_llm_personas_evaluation.py \
  --set temperature=0.1 --set timeout=300 --set refresh=true
```

The script accepts repeated `--set key=value` overrides. For instance,
`--set models='["mistral:latest","gemma2:2b"]'` limits the new run to two tags.
Treat the reported model status, prompt version, and actual settings as
part of the measurement whenever comparing runs.
