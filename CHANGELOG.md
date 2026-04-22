# Changelog

All notable changes to `humanbound-firewall` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/humanbound/humanbound-firewall/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/humanbound/humanbound-firewall/releases/tag/v0.2.0
[0.1.0]: https://github.com/humanbound/humanbound-firewall/releases/tag/v0.1.0
