# Heedle

*The layer where a person steers a model's change by hand, inside bounds that are checked.*

A [Moissanite Labs](https://github.com/Moissanite-Labs) project, and part of [ML Weft](https://github.com/Moissanite-Labs/weft).

![Status: early research](https://img.shields.io/badge/status-early%20research-orange)
![Code: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue)
![Docs: CC BY 4.0](https://img.shields.io/badge/docs-CC%20BY%204.0-lightgrey)

---

Weft's first design conversation described a loom in four layers. The third, the Heedle, is where a person and a model meet. A prototype in weft-loom built a first instrument for it. This repository builds the layer.

Heedle is an instrument for one loop, made by hand and checked by a program:

1. **Propose.** The instrument shows what a change would do before it is made.
2. **Steer.** You set a named change and how far to take it, in units of behaviour.
3. **Hold.** You mark what must not move.
4. **Check.** The change commits only through a gate on the weights and a claim about behaviour that one input can refute.
5. **Keep or discard.** A kept change joins a record that anyone can replay. A refuted one leaves the model where it was.

It starts from three ideas:

- **A knob is a claim with a handle.** Knobs are marked in units of behaviour. A named change comes with a number that says how much of the name the directions in use can carry, to first order. A committed change makes a claim, and the instrument looks for the one input that breaks it.
- **A forecast never refuses.** The instrument predicts how close a change comes to its limits and asks for a more deliberate gesture the closer it gets. Only the gate refuses, and only a refuted claim undoes a change.
- **Every session can be replayed.** A session is a hash-chained record. A separate program rebuilds every set of weights from it and re-checks every verdict.

## Two looms

| | Cloth | Text |
|---|---|---|
| The model | a point of the unit square to a colour | three characters to the next one |
| Weights | 467 | 4,700 |
| Where it came from | drawn from a seed | trained on this repository's own documents |
| What you see | a cloth: every pixel is an input, and the picture samples the square | a line of text per change, and the model's guesses after its commonest contexts |
| What a commit is checked on | a search of 9,600 points: a sampled reading | all 21,952 contexts: an exhaustive reading |

Both are toys. The text model's whole domain can be read. The cloth's can be seen at the resolution of the screen and searched. See [what it is not](#what-it-is-not).

## Quick start

You need Python 3.12 or newer with NumPy, and Node 20 or newer. Nothing touches the network.

```bash
git clone https://github.com/Moissanite-Labs/heedle
cd heedle
python3 -m http.server 8000
```

Open <http://localhost:8000/web/>. Press a treadle, watch the drawdown change, and beat the change in. Switch looms at the top.

Run what CI runs:

```bash
pip install numpy
python3 -m unittest discover -s tests -p 'test_*.py'    # the reference, both hosts
node --test web/test_heedle.mjs                         # the browser mirror, against the same vectors
python3 -m heedle verify drafts/sample-cloth.json       # replay a recorded session
python3 -m heedle verify drafts/sample-text.json
```

Copy a session out of the page with **Copy draft**, save it, and verify it the same way.

## Status

| Part | Path | What it is | Status |
|---|---|---|---|
| Core | [`heedle/core.py`](heedle/core.py), [`web/heedle.js`](web/heedle.js) | The mathematics for any host: shafts, treadles, pins, the proved bound, weight roots, claims, the loom | runs · tested in both languages against shared vectors |
| Hosts | [`heedle/hosts.py`](heedle/hosts.py), [`web/hosts/`](web/hosts) | `cloth/1` and `text/1` | runs · tested |
| Gate | [`web/vendor/invariant-gate.js`](web/vendor/invariant-gate.js), `core.verify_weight_update_safety` | The page runs weft's browser gate, copied unchanged and pinned by hash. The Python reference ports its unmasked path | runs · tested on weft's unmasked vectors, with the same deferrals as weft's browser test |
| Draft | [`docs/protocol.md`](docs/protocol.md), [`heedle/draft.py`](heedle/draft.py) | The record of a session and its verifier | runs · every rule of the specification has a tampered draft that is caught |
| Looms | [`web/`](web) | One page, two views | runs · checked by hand in one browser |
| Text model | [`hosts/text-v1.json`](hosts/text-v1.json) | The trained text host and the text it was trained on | fixed artefact, with its hashes |

**Not built:** signatures on drafts, a host for a real language model, a host for Weft's Plasm cells, sound, a second person at the loom, and any study with people. Each is an entry in the [roadmap](docs/roadmap.md).

## How to read this project

Heedle uses weaving names on purpose: *shaft, treadle, tie-up, heddle, shed, pick, fell*. To first order, the change a person makes is a product of three things, and a weaver's draft is a product of the same three. The names follow from that. They prove nothing. [docs/vocabulary.md](docs/vocabulary.md) says what each one names in the code.

The project holds itself to three rules. The first is weft's:

- A **stated** rule is not a **checked** rule. Documents link the test that checks a claim, or say it is not checked.
- A **passing test** is not a **proof**. [docs/theory.md](docs/theory.md) marks each statement as proved, tested on a toy, known, or a reading.
- A **forecast** is not a **verdict**. The instrument says which is which.

## What it is not

- **Not a way to steer a large model.** Both hosts are small enough to differentiate in a browser, and one is small enough to read whole. Whether any of this survives a model that is neither is the first open question in the roadmap.
- **Not evidence that people steer better with it.** No person has used it in a study.
- **Not a proof that a person was present.** A record holds the gesture the page saw.
- **Not a language model worth the name.** The text host knows three characters of context. It writes the shape of English and means nothing.

## What has been measured

One study exists, in weft-loom: [`experiments/heddle-knob-range-2026-11-17`](https://github.com/Moissanite-Labs/weft-loom/tree/main/experiments/heddle-knob-range-2026-11-17). It was made on the cloth host of the prototype this repository grew from. Seventeen predictions were fixed before the run. Sixteen were confirmed in each of two arms, and the record states the two that were not. Nothing has been measured on the text host.

## Documents

- [Charter](CHARTER.md): what Heedle is for, what it covers and what it does not
- [Theory](docs/theory.md): what the instrument computes and why, with each statement marked
- [Protocol](docs/protocol.md): the record of a session, and what a verifier checks
- [Hosts](docs/hosts.md): the contract, the two hosts, and how to add one
- [Vocabulary](docs/vocabulary.md): each word and what it names
- [Origins](docs/origins.md): the corpus's loom, and Spaceship IDE
- [Roadmap](docs/roadmap.md): what could grow from here, and the open questions

## Related repositories

| Repository | What it holds |
|---|---|
| [weft](https://github.com/Moissanite-Labs/weft) | The runtime. Heedle's page runs its browser gate unchanged. |
| [weft-loom](https://github.com/Moissanite-Labs/weft-loom) | The design note Heedle came from, the calibration theory and committed-behaviour-change design it rests on, and its first record. Studies of Heedle go there. |
| [weft-corpus](https://github.com/Moissanite-Labs/weft-corpus) | The design conversation that named the layer. |
| [Spaceship IDE](https://github.com/chuanqisun/spaceship) | By Chuanqi Sun. The four interaction ideas Heedle gives counterparts to. No code from it is used. |

## Contributing

Contributions that make a claim testable, or show that one is wrong, are the most valuable thing you can send. Start with [CONTRIBUTING.md](CONTRIBUTING.md). Every commit needs a sign-off (`git commit -s`). Please follow the [Code of Conduct](CODE_OF_CONDUCT.md), and report security problems privately as [SECURITY.md](SECURITY.md) describes.

## Licence

Code is under the [Apache License 2.0](LICENSE). Documents, the trained text model and its training text are under [CC BY 4.0](LICENSE-docs). See [NOTICE](NOTICE) for details and attribution.

## Disclosure

The first version of this repository was written by an AI agent session (Claude) at the request of the repository owner.
