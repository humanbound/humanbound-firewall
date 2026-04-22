# Roadmap

A snapshot of where `humanbound-firewall` is heading. This is a living
document — dates and scope may change. Open a [discussion](https://github.com/humanbound/humanbound-firewall/discussions)
or [issue](https://github.com/humanbound/humanbound-firewall/issues) to
weigh in.

The authoritative, continuously-updated roadmap lives at
[docs.humanbound.ai/defense/firewall/roadmap](https://docs.humanbound.ai/defense/firewall/).

## Now — shipping in the next release

- **Semantic detector plugins** — a unified interface so third parties can
  ship Tier 1 detectors as separate packages
- **Async `evaluate()`** — native `async def` support for hot paths
- **Richer benchmarks** — reproducible scripts + published numbers on a
  canonical corpus (PromptFoo + PyRIT + synthetic)

## Next — within the next two releases

- **Multi-modal inputs** — image + audio attack surface coverage
- **Streaming verdict** — early-exit for judge responses beyond the first
  token
- **Audit-trail export** — structured log output for SIEM integration

## Later — on the horizon, not committed

- **Adaptive thresholds** — per-agent learned consensus thresholds
- **Federated Tier 2 training** — coordinate across agent instances
  without centralising traffic
- **Official adversarial benchmark suite** — published, versioned, reusable

## Not doing

- Not becoming a general-purpose content moderator — Humanbound-firewall is
  scoped to prompt-injection / jailbreak / agent-scope-violation defense
- Not offering hosted inference — Humanbound's commercial platform offers
  that separately; the OSS package stays local-first
- Not shipping pre-trained weights with the source — models require separate
  licensing and ship via model hubs or the managed Humanbound Firewall service

## Release cadence

See [CONTRIBUTING.md](./CONTRIBUTING.md) for how changes ship and the semver
policy. The supported-version matrix is on the docs site.

---

_Humanbound is the trading name of AI and Me Single-Member Private Company,
incorporated in Greece._
