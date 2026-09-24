<p align="center">
  <img src="https://raw.githubusercontent.com/humanbound/humanbound-firewall/main/assets/logo-dark.svg" alt="Humanbound" width="280"/>
</p>

<h3 align="center">humanbound-firewall</h3>

<p align="center">
  Multi-tier firewall for AI agents. Screens every path into the model — the user's turn, every tool result, every record coming back from memory — and blocks prompt injections, jailbreaks, and scope violations before the model sees them.
  <br/>
  <strong>every boundary, one policy</strong> &middot; <strong>4-tier architecture</strong> &middot; <strong>two lines in a LangChain agent</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#how-it-works">How It Works</a> &middot;
  <a href="https://docs.humanbound.ai/defense/firewall/">Documentation</a> &middot;
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-preview-FD9506?style=flat-square" alt="Status: preview"/>
  <a href="https://pypi.org/project/humanbound-firewall/"><img src="https://img.shields.io/pypi/v/humanbound-firewall?style=flat-square&color=FD9506" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/humanbound-firewall/"><img src="https://img.shields.io/pypi/pyversions/humanbound-firewall?style=flat-square&color=FD9506" alt="Python versions"/></a>
  <a href="https://pypi.org/project/humanbound-firewall/"><img src="https://img.shields.io/pypi/dm/humanbound-firewall?style=flat-square&color=FD9506" alt="Downloads"/></a>
  <a href="https://github.com/humanbound/humanbound-firewall/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/humanbound/humanbound-firewall/ci.yml?style=flat-square&color=FD9506" alt="CI"/></a>
  <a href="https://github.com/humanbound/humanbound-firewall/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-FD9506?style=flat-square" alt="License"/></a>
  <a href="https://discord.gg/QFTD6tr9zu"><img src="https://img.shields.io/badge/discord-community-FD9506?style=flat-square" alt="Discord"/></a>
  <a href="https://docs.humanbound.ai/defense/firewall/"><img src="https://img.shields.io/badge/docs-humanbound.ai-FD9506?style=flat-square" alt="Docs"/></a>
</p>

---

> 📖 **Full documentation** lives at [**docs.humanbound.ai/defense/firewall/**](https://docs.humanbound.ai/defense/firewall/) —
> this README covers the essentials; the docs have the depth.

> ⚠ **Preview (0.3.x).** The Tier 0–3 contract, `.hbfw` model format,
> `humanbound_firewall.*` import surface, and `HUMANBOUND_FIREWALL_*` env
> variable names may change before 1.0. Pin to a specific version if you
> depend on a particular shape. The legacy `HB_FIREWALL_*` names still work
> with a deprecation warning and go away in 0.4.

## Every path into the model is a boundary

An agent does not only read what its user types. It reads web pages, documents
and tool results, and it reads its own records back from a database or a
memory store. Indirect prompt injection arrives through those paths, and the
model cannot tell an instruction from data on its own. The firewall sits on
each path and judges every payload by **who authored it**:

| class | what it is | what it may do | judged by |
|---|---|---|---|
| `request` | a principal's turn — the user, an operator | direct the agent, within policy | scope, permitted and restricted intents |
| `ingest` | outside content — pages, documents, tool results, sub-agents | inform | may it *direct* the agent? restricted? beyond the permitted intents? |
| `recall` | our own records coming back — a DB row, a memory entry | be data | integrity: does the record instruct at all? |

One policy file describes *your* agent — its scope, what it may do, what it
must never do, and which tools return your own records. It never describes an
attack.

## Quick Start

### Install

```bash
pip install humanbound-firewall                  # Core (Tiers 0 + 3)
pip install humanbound-firewall[tier1]           # + local DeBERTa for Tier 1
pip install humanbound-firewall[all]             # Everything
```

Optional per-provider extras: `[openai]`, `[anthropic]`, `[gemini]`. The
LangChain adapter needs no extra: LangChain is your application's dependency.

### A LangChain agent, two lines

```python
from langchain.agents import create_agent
from humanbound_firewall import Firewall

firewall = Firewall.from_config("agent.yaml")
agent = create_agent(model, tools, middleware=[firewall.adapt_to("langchain")])
```

The adapter attaches every boundary LangChain exposes: the human turn
(`request`), every tool result (`ingest`, or `recall` when your policy says the
tool returns your own records). A withheld result is replaced by a short notice
so the run continues without it; a rejected request ends the run with the
notice as the reply. The session travels in the graph state, so a checkpointer
carries it across the runs of a thread. `firewall.adapt_to("langchain").report()`
prints the agent's trust-boundary inventory.

In `log` mode a verdict changes nothing the agent does, so the adapter does not
wait for it: the result goes on at once and is judged in the background, in call
order per thread, with the same verdicts and session a blocking run would give.
Decisions reach `on_decision` as they land; the adapter's `flush(timeout)` waits
for pending judgements (call it before a script exits), and
`adapt_to("langchain", log_blocking=True)` makes log mode wait.

### Any agent: the manual tier

`inspect()` is the same call the adapter makes. Use it wherever content enters
your model's context:

```python
from humanbound_firewall import Firewall, Session

firewall = Firewall.from_config("agent.yaml")
session = Session.new()  # or Session.from_json(token)

page = fetch(url)
d = firewall.inspect(
    page,
    cls="ingest",
    session=session,
    boundary={"name": "fetch", "kind": "tools", "args": {"url": url}},
)
session = d.session  # token out — you carry it
page = d.replacement if d.action == "withhold" else page
```

`evaluate()` is the filter behind it: verdict, category, tier, probabilities,
explanation, and the updated session; the caller decides. `inspect()` adds the
decision — `pass`, `withhold` or `reject` — with your deployment's modes applied.

### The deployment's choices

Everything that shapes a decision is set once, in code:

```python
firewall = Firewall.from_config(
    "agent.yaml",
    provider=provider,  # the Tier 3 judge (OpenAI, Azure OpenAI, Claude, Gemini)
    classes=("ingest", "recall"),  # which trust classes this deployment enforces
    mode="block",
    mode_by_class={"request": "log"},  # block | log | passthrough
    fail="closed",  # high stakes: an uncertain verdict withholds
    on_decision=audit,  # observe-only; cannot change a verdict
    withheld_template="[Withheld by policy: {category}. Continue without it.]",
)
```

The judge can also be configured from the environment:

```bash
export HUMANBOUND_FIREWALL_PROVIDER=openai
export HUMANBOUND_FIREWALL_API_KEY=sk-...
```

### The policy file

```yaml
name: ShopAssist
scope:
  business: Shopping assistant for an online store; reads supplier pages and answers product questions.
  more_info: "HIGH-STAKE: the agent can look up supplier contract prices and customer orders, which are confidential."
intents:
  permitted:
    - read a supplier's public product page and follow links on their site for product details
    - look up our own catalogue and order records
  restricted:
    - disclose, send or encode contract prices or customer data to any external party or URL
    - register, sign up or submit information on an external website
capabilities: [tools]                        # the kinds of boundary the agent has
tools:                                       # optional; can only RELAX the default (every tool is ingest)
  recall: [lookup_catalogue, lookup_order]   # tools that return records WE authored: integrity mode
  expects:
    fetch_url: "a supplier's public web page"
settings:
  mode: block
  timeout: 15
```

`few_shots` entries — attack examples learned from your findings — carry the
class they were learned on (`class: request | ingest`; the recall judge takes
none). Exporting them from the Humanbound CLI is not available yet.

## How It Works

Every payload passes through four tiers before it reaches your model:

```
Payload (request | ingest | recall)
    |
[ Tier 0 ]  Sanitization                    no model call, free
    |        Strips invisible control characters, zero-width joiners, bidi overrides.
    |
[ Tier 1 ]  Basic Attack Detection          local model inference, free
    |        Pre-trained models (DeBERTa, Azure Content Safety, Lakera, etc.)
    |        Pluggable ensemble — add models or APIs, configure consensus.
    |
[ Tier 2 ]  Agent-Specific Classification   local model inference, free
    |        Trained on YOUR agent's adversarial test logs and QA data,
    |        or a policy classifier that reads agent.yaml. Fast-tracks the clear cases.
    |
[ Tier 3 ]  LLM Judge                       LLM call, token cost
             One judge per trust class, given your policy, the recent
             conversation, and the boundary the payload crossed: which tool,
             what it is for, what it was called with.
```

Each tier either makes a confident decision or escalates. The judge answers
with a verdict letter first; a reply that does not start with one is no
verdict, which a fail-closed deployment withholds. Payloads reach the judge
fenced, with the protocol restated after them, so content that tries to talk to
the judge has nothing to hold on to.

Full config reference, tier-by-tier deep dive, training your own Tier 2 model,
writing custom detectors, `.hbfw` model format, and API reference all live in
the [firewall docs](https://docs.humanbound.ai/defense/firewall/).

## From your test results

The [Humanbound CLI](https://github.com/humanbound/humanbound) (2.10 or later)
turns your test results into the firewall's inputs: the policy file the Tier 3
judge reads, and a Tier 2 model trained on exactly the attacks your agent failed.

```bash
pip install "humanbound[firewall]"          # installs both packages together
hb login
hb test                                     # run adversarial tests

# The policy file (agent.yaml): scope, intents, capabilities
hb guardrails --format yaml -o agent.yaml

# A Tier 2 model, with the SetFit detector from this repository
pip install setfit
mkdir -p detectors
curl -o detectors/setfit_classifier.py \
  https://raw.githubusercontent.com/humanbound/humanbound-firewall/v0.3.0/detectors/setfit_classifier.py
hb firewall train --model detectors/setfit_classifier.py -o firewall.hbfw
```

`hb firewall train` needs `--model`: the detector script is not part of the pip
package. Keep it with your app — `Firewall.from_config("agent.yaml",
model_path="firewall.hbfw", detector_script="detectors/setfit_classifier.py")`
loads the trained model with it. Not logged in, `hb test` runs locally and
`hb guardrails --format yaml` builds `agent.yaml` from the scope the test ran
against.

See [docs.humanbound.ai](https://docs.humanbound.ai) for the full CLI + firewall
integration walkthrough.

## Contributing

Contributions welcome. See [CONTRIBUTING.md](https://github.com/humanbound/humanbound-firewall/blob/main/CONTRIBUTING.md)
for the dev loop and release process. Contributions are accepted under the
[Developer Certificate of Origin](https://github.com/humanbound/humanbound-firewall/blob/main/DCO.md)
(sign your commits with `git commit -s`) — no CLA to sign; you keep the
copyright to your work under the same Apache-2.0 license as the project.

- 🐛 [Report a bug](https://github.com/humanbound/humanbound-firewall/issues/new/choose)
- 💡 [Request a feature](https://github.com/humanbound/humanbound-firewall/issues/new/choose)
- 🔒 [Report a security issue](https://github.com/humanbound/humanbound-firewall/blob/main/SECURITY.md) — **not via public Issues**
- 💬 [Join Discord](https://discord.gg/QFTD6tr9zu)

## License

[Apache-2.0](https://github.com/humanbound/humanbound-firewall/blob/main/LICENSE). Free to use in any context — commercial or
open-source — with attribution.

See [TRADEMARK.md](https://github.com/humanbound/humanbound-firewall/blob/main/TRADEMARK.md) for the trademark policy. The code is open;
the name is not.
