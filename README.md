# assay-harness

Open evaluation harness for the [Agentsia Labs](https://labs.agentsia.uk) benchmark series.

`assay-harness` is the scoring pipeline behind every Agentsia Labs release. It loads a scenario dataset, runs it through one or more model runners (frontier APIs, open-weights inference endpoints, local vLLM), evaluates outputs against a published rubric, and writes versioned, reproducible results.

Anyone can use this harness to reproduce Agentsia Labs numbers, or to score their own models against our published datasets, without involving Agentsia.

## Status

**v0.3.0 · 2026-04 · Anthropic + OpenAI adapters live.** The public surface (types, CLI shape, runner interface, rubric contract, output format) is stable. Anthropic Messages API and OpenAI Chat Completions adapters are implemented; Google, Hugging Face, and vLLM are still stubs. The full cross-provider harness ships alongside the inaugural Agentsia Labs benchmark, Assay-Adtech v1, targeted Q2 2026.

## Scope

Evaluation execution only. This repo does not contain:

- The synthetic scenario-generation pipeline that authors Assay datasets. That lives inside Modelsmith.
- The post-training pipeline used to build Agentsia specialist models.
- Any customer or commercial Modelsmith code.

The separation is deliberate. Assay datasets, the harness, and the leaderboards are fully open. The generator and the training pipeline are the commercial surface of the Agentsia platform. Both are referenced by the methodology page at [labs.agentsia.uk/methodology](https://labs.agentsia.uk/methodology).

## Install

```bash
pnpm install
```

Requires Node 22 or later.

## Run

```bash
# List the bundled sample scenarios
pnpm assay list examples/scenarios

# Score a dataset against a single runner
pnpm assay run \
  --dataset examples/scenarios \
  --runner stub:echo \
  --out runs/local.json

# Score across multiple runners
pnpm assay run \
  --dataset examples/scenarios \
  --runner anthropic:claude-opus-4-7 \
  --runner openai:gpt-6 \
  --runner google:gemini-3-pro \
  --out runs/cross.json
```

Each runner emits a `ModelResponse` per scenario. A `Rubric` attached to each scenario converts the response to a `Score`. The aggregator collapses scores into per-axis summaries and a weighted composite. Output is serialised as JSON, versioned against the dataset SHA and the harness release tag.

## Concepts

| Term | Meaning |
|---|---|
| **Scenario** | One test case: prompt input, expected rubric, capability axis label, and scenario metadata. Serialised as JSON. |
| **Runner** | A provider-specific adapter that submits a scenario prompt and returns a `ModelResponse` with version, timestamp, latency, and generation settings. |
| **Rubric** | The scoring contract for a scenario. Three kinds: programmatic (structural checker), LLM-as-judge (reference-matched), human (panel review). |
| **Score** | A 0-to-1 value for one runner on one scenario on one axis, with optional rationale. |
| **Axis** | A capability dimension published for a benchmark (e.g. bid-shading judgement, pre-bid MFA filtering, RTB-payload parsing). |
| **Composite** | Weighted average across axes. Weights are published alongside the release with a rationale. |
| **RunRecord** | Top-level output: the dataset version, the runners evaluated, every `Score`, and every `ModelAggregate`. |

## Runners (planned)

| Runner id | Provider | Implementation status |
|---|---|---|
| `anthropic:*` | Anthropic Messages API | implemented (v0.2) |
| `openai:*` | OpenAI Chat Completions API | implemented (v0.3) |
| `google:*` | Google Gemini API (no grounding) | stub |
| `hf:*` | Hugging Face Inference endpoint | stub |
| `vllm:*` | Local vLLM server | implemented (v0.4) |
| `stub:echo` | Returns the prompt verbatim; deterministic; used for tests | implemented |
| `stub:empty` | Returns the empty string; deterministic; used for failure-mode tests | implemented |

Each runner discloses: `provider`, `model`, `version`, `temperature`, `systemPrompt`, `accessedAt`, `latencyMs`. These travel with every `ModelResponse` and are serialised into the `RunRecord`.

## Rubrics (planned)

- **Programmatic.** Deterministic checker; the correct output is structurally decidable (schema match, value equality, computed predicate). Implemented as a small TypeScript function.
- **LLM-as-judge.** A pinned judge model scores the response against a reference. Judge model id is disclosed in the release; inter-judge agreement is published when multiple judges are used.
- **Human.** Panel-reviewed scoring surfaced through a thin annotation interface. Not in v0.

## Reproducibility

Every published Assay release produces:

- The scenario dataset at a specific git tag (Apache 2.0)
- The harness version (git tag on this repo)
- The reproduction command (a single `assay run` invocation)
- The raw `RunRecord` JSON with per-response outputs
- The leaderboard computed from the `RunRecord`

If a user cannot reproduce a published number within reported variance by running the command on the published artefacts, that is a bug. File an issue.

## Variance and seeds

Every score is run three times with different sampling seeds (temperature 0 where permitted; 0.2 otherwise). Variance is reported alongside the mean. Small deltas between runners in high-variance regimes are explicitly flagged in the release, not treated as ranked.

## Development

```bash
pnpm install
pnpm typecheck
pnpm test
pnpm build
```

CI runs on GitHub Actions for every PR: typecheck, test, and build.

## Contributing

Issues and pull requests welcome. For substantive changes (new runner, new rubric type, scoring-logic changes) please open an issue first so we can discuss the shape. For typos, documentation, and small fixes please open a PR directly.

## Licensing

Apache 2.0. See `LICENSE`.

## Citation

If you cite Agentsia Labs numbers in your own work, please cite the specific benchmark release (e.g. `Assay-Adtech v1.0`) rather than this repo. The repo is the tool; the benchmark is the claim.

## Related

- [labs.agentsia.uk](https://labs.agentsia.uk) · the Labs surface, methodology, roadmap, and leaderboards
- [agentsia.uk](https://agentsia.uk) · Agentsia, the specialisation control plane for enterprise model fleets
