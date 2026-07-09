# Changelog

All notable changes to `humanbound-firewall` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
