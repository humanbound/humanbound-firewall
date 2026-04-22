# Contributing to humanbound-firewall

Thanks for considering a contribution. This document covers the essentials —
for extended guidance, see [docs.humanbound.ai/community/contributing](https://docs.humanbound.ai/community/).

## Quick start

```bash
git clone https://github.com/humanbound/humanbound-firewall.git
cd humanbound-firewall
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,tier1]'
pre-commit install
pytest
```

That sequence gets you a working dev environment with the test suite, linter,
and formatter wired up.

## Filing issues

Bugs, feature requests, and questions all live in [GitHub Issues](https://github.com/humanbound/humanbound-firewall/issues).
Use the provided templates:

- **Bug report** — include `humanbound-firewall` version, Python version, and a
  minimal reproduction
- **Feature request** — describe the problem first, then the proposed solution

**Do not file security issues publicly.** See [SECURITY.md](./SECURITY.md) for
the private disclosure channel.

## Contributor License Agreement (CLA) — required

Every external contribution must be covered by the
[Humanbound Contributor License Agreement](./CLA.md). The CLA gives
Humanbound the operational flexibility to evolve the project (including
offering managed services on the Humanbound Platform) while preserving Your
right to use Your own contributions elsewhere and Your authorship in the
project's git history.

The first time you open a pull request, the CLAAssistant bot will comment
with a one-line instruction to sign. Sign once and all your future
contributions are covered.

## Change workflow

1. Fork the repository and create a branch off `main`
2. Make your changes — keep them focused (one concern per PR)
3. Add or update tests
4. Ensure `pytest`, `ruff check`, and `mypy` pass locally
5. Update [CHANGELOG.md](./CHANGELOG.md) under the `[Unreleased]` section
6. Open a pull request using the template

### Code style

- Formatter and linter: `ruff` (run via `pre-commit`)
- Type checker: `mypy` (see `pyproject.toml` for configuration)
- Every new `.py` file gets the SPDX header:
  ```
  # SPDX-License-Identifier: Apache-2.0
  # Copyright (c) 2024-2026 Humanbound
  ```

### Tests

- Every new feature needs a test
- Every bug fix needs a regression test
- Tests run against Python 3.10, 3.11, and 3.12 in CI
- Keep bare `import humanbound_firewall` fast — there is a regression test
  (`tests/test_import_time.py`) that guards the cold-import budget

## How changes ship

Maintainers cut releases on a rolling basis, not on a fixed cadence.

| Step | Who | What |
|---|---|---|
| PR review | Maintainer | Reviews code, tests, CHANGELOG |
| Merge to `main` | Maintainer | Squash merge |
| Tag `vX.Y.Z` | Maintainer | Triggers `release.yml` |
| Publish to PyPI | CI via Trusted Publishing | No tokens, sigstore-signed |
| GitHub Release | CI | Created from `CHANGELOG.md` entry |

Versioning follows [semver](https://semver.org). See
[docs.humanbound.ai/community/release-process/](https://docs.humanbound.ai/community/)
for the current cadence and supported-version matrix.

## Community

- **Discord** — [discord.gg/gQyXjVBF](https://discord.gg/gQyXjVBF) for questions
  and discussion
- **Discussions** — on the GitHub repo, for longer-form topics
- **Docs** — [docs.humanbound.ai](https://docs.humanbound.ai)

## Code of Conduct

Participation is governed by our [Code of Conduct](./CODE_OF_CONDUCT.md).
Violations can be reported privately to
[conduct@humanbound.ai](mailto:conduct@humanbound.ai).

---

_Humanbound is the trading name of AI and Me Single-Member Private Company,
incorporated in Greece._
