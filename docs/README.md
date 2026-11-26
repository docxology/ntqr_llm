# NTQR LLM documentation

The study evaluates binary LLM-persona votes using NTQR and compares inferred
accuracies against authored scenario labels. The retained votes and their
analysis support offline reproduction.

| Guide | Use it to |
|---|---|
| [Study overview](study.md) | Understand the research question, results, and empirical limits. |
| [Manuscript and supplement](manuscript/README.md) | Read the full study, derivations, and results. |
| [Setup and verification](VERIFICATION.md) | Install the locked environment and audit the recorded votes without Ollama. |
| [Experiment configuration](configuration.md) | Find the persona, scenario, model, and collection settings. |
| [Rendering pipeline](rendering_pipeline.md) | Generate the numbered sections, then render and validate PDF and HTML. |
| [Manuscript authoring](manuscript_guide.md) | Edit the canonical templates, metadata, and figure references. |
| [Architecture](architecture.md) | Trace the workflow from prompts or recorded votes through the evaluators and renderer. |
| [Recovery validity](recovery_validity.md) | Interpret the physical-solution audit and artifact manifests. |
| [Research integrity](research_integrity.md) and [claim registry](research_map.json) | Trace the evidence, citations, and inference boundaries. |
| [Testing philosophy](testing_philosophy.md) | Understand the test layers and fixture policy. |
| [NTQR 0.9 migration](ntqr09.md) | Check vendored source provenance and local changes. |
| [Runtime profiling](runtime_profiling.md) | Run bounded profiles and interpret the archived benchmark receipts. |
| [Essays](essays/README.md) | Read the conceptual background. |
| [Maintenance backlog](TODO.md) | Find pending engineering work. |

The [analysis artifact index](../outputs/README.md) describes the saved
outputs and data sidecars. The [delivery artifact index](../output/README.md)
describes the renderer staging and publication files. The published version
record is [Zenodo 20498700](https://zenodo.org/records/20498700).
