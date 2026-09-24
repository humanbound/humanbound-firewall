# Changelog

All notable changes to `humanbound-firewall` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **The `hb_firewall` alias and the `HB_FIREWALL_*` environment names are
  removed in 0.4**, not 0.3 as the deprecation messages said; 0.3.x keeps
  both. The messages now say 0.4.
- **`settings.session_window` and `settings.temperature` warn.** Both were
  parsed and never used — the judge runs at temperature 0 and sees every turn
  it is given. Setting either now raises a `DeprecationWarning`; both are
  removed in 0.4.

### Docs
- **README:** the Humanbound CLI section exports the policy file
  (`hb guardrails --format yaml`) and trains Tier 2 with the required
  `--model` and the SetFit detector; log mode's background judging
  (`flush()`, `log_blocking`) is described; few-shot examples are no longer
  said to be exported by the CLI.

## [0.3.0] — 2026-09-24

Every path into the model's context is a boundary. This release makes the
firewall judge a payload by who authored it, adds a gateway that enforces the
deployment's choices, a session the caller carries, and a LangChain adapter
that attaches every boundary with two lines.

### Added
- **Trust classes.** `evaluate(..., cls=)` takes `request` (a principal's
  turn), `ingest` (outside content: tool output, pages, documents,
  sub-agents) or `recall` (the agent's own records coming back). A
  conversation ending with a `{"role": "tool"}` message is ingest. `CLASSES`
  is exported. Tier 3 gets one judge per class: `prompts/judge.txt` (request,
  unchanged), `prompts/judge_ingest.txt` (untrusted data) and
  `prompts/judge_recall.txt` (integrity). `Category.INTEGRITY` names a recall
  block. Class-aware Tier 2 classifiers (`supports_class = True`) are
  consulted for single-shot ingest and recall payloads; classifiers written
  before trust classes are called exactly as before.
- **`Firewall.inspect()` and `Guard`.** The gateway call: everything
  `evaluate()` reports plus the action (`pass`, `withhold`, `reject`) and the
  replacement text, with the deployment's choices applied. `from_config()`
  and `Firewall()` take `classes`, `mode`, `mode_by_class`, `fail`
  (`open` | `closed`), `on_decision` (observe-only) and `withheld_template`
  (`{category}`, `{explanation}`, `{cls}`, `{boundary}`). An engine failure
  is the agent's failure: `evaluate()` raises, `inspect()` resolves it by the
  fail mode. `Decision` is exported.
- **`Session`.** A pure, serialisable value the caller carries — token in,
  token out — built from verdict-time facts only. `evaluate()` and
  `inspect()` take `session=` and return the updated value; a block earlier
  in the thread elevates the posture, which tightens a Tier 2 classifier that
  exposes `tightened()` one notch. `Session.merge()` joins branches that
  recorded in parallel.
- **Policy file.** `capabilities` (the platform's vocabulary: `tools`,
  `memory`, `inter_agent`, `reasoning_model`) and an optional whitebox
  `tools:` block — `recall: [...]`, `class: {tool: ingest|recall}`,
  `expects: {tool: "..."}` — that can only relax the default (every tool is
  ingest). `AgentConfig.class_of(tool)`. `few_shots` entries carry the class
  they were learned on (`class: request | ingest`, default `request`); the
  recall judge takes none.
- **`Firewall.adapt_to(framework)`** and a LangChain adapter
  (`humanbound_firewall.integrations.langchain.HumanboundFirewallMiddleware`):
  `before_agent` starts the run and pins the request, `before_model` judges
  the human turn as `request`, `wrap_tool_call` judges every tool result as
  `ingest` or `recall` per policy and replaces a withheld result with the
  template notice, tool_call_id intact. The session lives in the graph state
  under `humanbound_session` with a reducer for parallel tool calls.
  `boundaries` is the trust-boundary inventory; `report()` prints it. Adapters
  are imported lazily. A conformance test drives one scripted thread through
  the manual tier and the adapter against the same fake engine.
- **Log mode off the agent's path.** In log mode a verdict changes nothing the
  agent does, so the LangChain adapter no longer waits for it: the tool result
  (or the human turn) goes on at once and the same payload is judged in the
  background, with the window, boundary and session captured at that moment.
  A thread's background judgements run one at a time in call order, each on
  the session the previous one produced, so verdicts and session match a
  blocking run; threads are judged in parallel. Decisions reach `on_decision`
  as they land; the session is written back at the next model turn and in
  `after_agent`; `flush(timeout)` waits for pending judgements.
  `adapt_to("langchain", log_blocking=True)` restores waiting. A tool
  boundary now carries its `tool_call_id`.
- **The boundary in the judge's context.** The ingest and recall judges
  receive a BOUNDARY section — the tool's own description, the policy's
  `expects`, and what the tool was called with — and use it generically.
- `evaluate()` accepts `window=` (the recent transcript, OpenAI format) and
  `boundary=` alongside a plain payload. `EvalResult.wait_explanation()`
  waits for the streamed Tier 3 explanation. `EvalResult.session`.

### Changed
- **The ingest judge.** Content that only informs — data, notices, errors,
  empty results, pages about something else — is never blocked, whatever its
  subject; directive detection is exhaustive and the information around a
  directive never excuses it; the judge evaluates this payload's implied
  action and never speculates about a linked page; provenance compares where
  a payload sends the agent with where it came from; `D` is reserved for
  unreadable content.
- **The recall judge.** Values are never a violation, however sensitive; a
  record is expected to be the answer to the call it was made with; empty
  results and errors are data; a quoted conversation is data unless the
  record addresses the agent now; a polite request to the assistant is an
  instruction.
- **Verdict extraction.** The verdict is the letter the judge's reply starts
  with, read to its word boundary; a letter inside a word — or inside the
  payload's own text, from a judge captured into echoing it — never counts.
  A reply that opens with a preamble is asked once more, tersely; a second
  violation is no verdict, with the explanation quoting the reply. Payloads
  reach the judge fenced, with the protocol restated after them.
- Every OpenAI and Azure OpenAI model is sent the parameters it accepts:
  `max_completion_tokens` with headroom for reasoning models, and
  `temperature` retried away when a model rejects it. No model-name lists.
- The Tier 3 judge returns at once when a stream ends without a verdict.
- `settings.risk_tolerance`, undocumented and unused, is removed; declare
  stakes the documented way, a `HIGH-STAKE:` / `MEDIUM-STAKE:` /
  `LOW-STAKE:` marker in `scope.more_info`.
- Example verdicts and lists in the judge prompts use neutral wording.

### Removed
- **`Metrics.record_error()`, `Metrics.errors` and the `"errors"` key of
  `Metrics.to_dict()`.** Nothing ever called `record_error()`: a judge timeout,
  API error or broken stream is returned as a `review` verdict and counted as
  one, so `errors` was always 0. A counter that reads 0 during an outage is
  worse than none.

## [0.2.3] — 2026-09-14

### Changed
- **Contribution policy: CLA replaced by DCO.** External contributions no
  longer require signing the Humanbound Contributor License Agreement.
  Contributions are now accepted under the Developer Certificate of Origin
  v1.1 (see `DCO.md`) — sign commits with `git commit -s`. Contributors keep
  their copyright; contributions are licensed inbound = outbound under
  Apache-2.0. `CLA.md` is removed and a `dco.yml` workflow now checks
  `Signed-off-by` trailers on every pull request.
- `CONTRIBUTING.md` gains an explicit third-party license policy: vendored
  code must be permissively licensed (Apache-2.0/MIT/BSD/ISC); GPL, AGPL,
  SSPL, and BSL code cannot be accepted.
- **Discord links now point at the `#start-here` invite** (#19).
  `CONTRIBUTING.md` and the new-issue chooser linked `discord.gg/gQyXjVBF`,
  which no longer resolves; `README.md` and `pyproject.toml` used a live but
  different invite. All four now use `discord.gg/QFTD6tr9zu`, matching
  www.humanbound.ai. The PyPI sidebar link updates on the next release.
- `LICENSE` restored to the verbatim Apache-2.0 text. Sections 6 and 9 had
  diverged from the canonical wording and the appendix was missing; the file
  now matches apache.org/licenses/LICENSE-2.0 exactly, apart from the
  appendix copyright line. The license grant is unchanged — the project was
  and remains Apache-2.0.

### Added
- `NOTICE` file per Apache-2.0 section 4(d).

### Fixed
- `__version__` in `humanbound_firewall/__init__.py` read `0.2.1` while
  `pyproject.toml` declared `0.2.2`, so the installed package misreported
  its own version. Both now agree.

### Security
- **`.hbfw` files can no longer execute code on load.** `load_hbfw()` used
  `numpy.load(..., allow_pickle=True)`, so a crafted model ran arbitrary code
  on load — before the Tier 2 detector-script check. Loading is now
  `allow_pickle=False`. No legitimate model is affected (weights are numeric
  arrays, strings, or `np.void` blobs); only object-dtype arrays are rejected.
- **Arbitrary file write via the reference detector's manifest.**
  `detectors/setfit_classifier.py` wrote manifest files under a temp dir
  without constraint, so an absolute path or `..` escaped it. Absolute and
  escaping paths are now rejected.
- Pinned every GitHub Action used in the workflows to a full commit SHA
  (with a version comment), so a moved tag cannot inject code between
  Dependabot updates. `pypa/gh-action-pypi-publish` had been tracking
  `release/v1` — a branch — in the job that holds PyPI publishing rights.
  No action changed version: each SHA is what its tag resolved to at the
  time of the change.

## [0.2.2] — 2026-07-09

### Removed
- **`hb-firewall` transitional stub retired.** All `hb-firewall` releases on
  PyPI were yanked on 2026-07-09 (past the 2026-06-20 transition window
  announced in 0.2.0), with the yank reason pointing users to
  `pip install humanbound-firewall`. The stub sources (`compat/hb-firewall`)
  and its `build-stub` / `publish-stub` release-workflow jobs are removed, so
  future releases no longer publish the deprecated package.

### Fixed
- **PyPI listing**: replaced every repo-relative link in the README (CONTRIBUTING,
  CLA, SECURITY, LICENSE, TRADEMARK) with absolute GitHub URLs — they returned 404
  when rendered on pypi.org. Also fixed the garbled CLA sentence in the
  Contributing section.
- **PyPI logo**: the `<picture>` theme-switching block is stripped by PyPI's HTML
  sanitizer, leaving a broken relative image. The README now uses a single absolute
  `<img>` pointing at the self-contained logo variant that reads on both PyPI themes.

### Added
- Package metadata: `Programming Language :: Python :: 3.13` and
  `Topic :: Scientific/Engineering :: Artificial Intelligence` classifiers, and a
  `Discord` entry in `[project.urls]` so PyPI shows the community link in the sidebar.

### Changed
- **README refresh**: quick start now leads with the code (Tiers 0–2 run
  locally and free; provider env vars only needed for the Tier 3 LLM Judge),
  and the CLI section is reframed as "Train guardrails from your test
  results". Contributing/CLA wording cleaned up.
- **Removed unsourced performance claims** from the README tagline, tier
  diagram, and package summary ("sub-millisecond latency", per-tier timings,
  "~85% of prompt injections", "~10-15% of traffic"). Until we publish
  benchmarks, performance language is qualitative and architecture-grounded
  only: local tiers are free with no LLM call; the Tier 3 judge is the only
  tier with token cost and is invoked only on escalation.
- Package summary now reads "Multi-tier firewall for AI agents — blocks prompt
  injections, jailbreaks, and scope violations; fast local tiers screen every
  request, only uncertain cases reach an LLM judge."

## [0.2.1] — 2026-05-12

### Fixed
- **Tier 1 silent fail-open**: `AttackDetector` now surfaces failures
  instead of returning `0.0` (which the ensemble couldn't distinguish from
  a genuine "no attack" score). When a detector's HTTP call raises, returns
  a non-200 response, or its local model errors, `score()` returns `None`.
  The ensemble skips `None` results and emits a `logging.WARNING` so
  operators see that the firewall is running with reduced coverage. Failed
  detectors no longer cast a silent "safe" vote. (#8)

### Changed
- `AttackDetector.score()`, `_score_api()`, and `_score_local()` now return
  `float | None` instead of `float`. Code that arithmetically combined
  detector scores must handle the new sentinel. `_score_local` still
  re-raises `ImportError` so users get the actionable "install the [tier1]
  extra" hint. (#8)

## [0.2.0] — 2026-04-21

### Changed
- **License**: relicensed from AGPL-3.0-only (with a separate commercial
  option) to **Apache-2.0**. External contributions are now accepted under
  the Humanbound Contributor License Agreement (see `CLA.md`) so the project
  can continue to evolve and be offered through commercial channels.
- **Renamed package**: `hb-firewall` → `humanbound-firewall`. The old name
  remains on PyPI as a transitional meta-package through at least 2026-06-20,
  emitting a `DeprecationWarning` on import.
- **Renamed module**: `hb_firewall` → `humanbound_firewall`. A legacy
  `sys.modules` alias keeps `import hb_firewall` working for 0.2.x so old
  `.hbfw` pickled models still load. The alias will be removed in 0.3.
- **Copyright attribution**: LICENSE now names AI and Me Single-Member
  Private Company (also known as Humanbound) as the copyright holder,
  reflecting the company's current corporate identity.
- Switched to brand-only `Humanbound` in per-file SPDX headers.

### Added
- **Lazy imports** — bare `import humanbound_firewall` now stays under
  ~200 ms by deferring `Firewall`, `HBFW`, and LLM-provider loading until
  first attribute access.
- `py.typed` marker (PEP 561) — downstream users of type checkers now see
  the package's type hints.
- **Actionable import errors** for missing optional dependencies. Missing
  `torch` / `transformers` / `openai` / `anthropic` / `google-generativeai`
  now raises an `ImportError` telling the user which extra to install.
- **OSS hygiene documents**: `SECURITY.md`, `CODE_OF_CONDUCT.md`,
  `CONTRIBUTING.md`, `CLA.md`, `TRADEMARK.md`, `ROADMAP.md`.
- **GitHub automation**: CI matrix (Python 3.10 / 3.11 / 3.12), release
  workflow with PyPI Trusted Publishing (OIDC) and sigstore attestations,
  issue/PR templates, dependabot, CLAAssistant configuration.
- **Dev tooling**: `.pre-commit-config.yaml`, ruff + mypy configuration in
  `pyproject.toml`, cold-import regression test.
- `examples/quickstart.py` — runnable version of the README's quickstart.
- Transitional stub package `hb-firewall==0.2.0` published alongside the new
  name so existing users get a clear deprecation signal.

### Deprecated
- The `hb-firewall` PyPI name. Install `humanbound-firewall` instead. The
  stub will be yanked on or after 2026-06-20.
- The `hb_firewall` module path. Import from `humanbound_firewall` instead.
  The alias will be removed in 0.3.
- Environment variables renamed from `HB_FIREWALL_*` to `HUMANBOUND_FIREWALL_*`
  (`HUMANBOUND_FIREWALL_PROVIDER`, `HUMANBOUND_FIREWALL_API_KEY`,
  `HUMANBOUND_FIREWALL_MODEL`, `HUMANBOUND_FIREWALL_ENDPOINT`,
  `HUMANBOUND_FIREWALL_API_VERSION`). The legacy `HB_FIREWALL_*` names still
  work for 0.2.x with a `DeprecationWarning`; they are removed in 0.3.

## [0.1.0] — 2026-03-26

### Added
- Initial public release as `hb-firewall`.
- Four-tier firewall architecture: sanitization, pre-trained attack
  detection ensemble, agent-specific trained classifiers, and LLM-as-a-judge.
- Pluggable Tier 1 ensemble with configurable consensus; supports local
  HuggingFace models and remote API endpoints.
- `.hbfw` portable model format for Tier 2 classifiers.
- Default SetFit-based `AgentClassifier`.
- Support for OpenAI, Azure OpenAI, Anthropic Claude, and Google Gemini as
  Tier 3 judges.
- YAML-based agent configuration.

[Unreleased]: https://github.com/humanbound/humanbound-firewall/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/humanbound/humanbound-firewall/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/humanbound/humanbound-firewall/releases/tag/v0.2.0
[0.1.0]: https://github.com/humanbound/humanbound-firewall/releases/tag/v0.1.0
