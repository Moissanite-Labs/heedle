# Contributing to Heedle

Heedle is early. The most useful contributions make a claim testable or show that one is wrong. A failing test that exposes a gap is a welcome pull request.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ground rules for evidence

These matter more here than style does. They are weft's rules, and the [charter](CHARTER.md) repeats them.

1. **A stated rule is not a checked rule.** If a document says the instrument enforces something, link the test that checks it. Otherwise say it is not checked.
2. **A passing test is not a proof.** [docs/theory.md](docs/theory.md) marks each statement as proved, tested, known or a reading. Keep the marks honest when you change a statement.
3. **Say what is toy.** Every host here is small enough that its whole domain can be read. Do not describe a result on one as a law.
4. **Nothing live.** Tests and tools must not need keys or network access beyond installing packages.
5. **Keep the two languages together.** A change to the mathematics goes into `heedle/` and `web/heedle.js` in the same pull request, with the vectors rebuilt.

## Where things live

| You have | Put it in |
|---|---|
| A change to the mathematics | `heedle/core.py`, `web/heedle.js`, and a vector in `vectors/` |
| A new host | `heedle/hosts.py`, `web/hosts/`, and a section in [docs/hosts.md](docs/hosts.md) |
| A change to the page | `web/` |
| A change to the draft format | [docs/protocol.md](docs/protocol.md) first, then `heedle/draft.py` and `web/heedle.js` |
| An experiment, a pilot or a result | [weft-loom](https://github.com/Moissanite-Labs/weft-loom), as a record. Link it from [docs/theory.md](docs/theory.md) |
| An idea that is not yet code | [docs/roadmap.md](docs/roadmap.md), or a note in weft-loom |

## Setup

You need Python 3.12 or newer with NumPy, and Node 20 or newer.

```bash
git clone https://github.com/Moissanite-Labs/heedle
cd heedle
python3 -m venv .venv && . .venv/bin/activate
pip install numpy
```

Run what CI runs:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
node --test web/test_heedle.mjs
python3 -m heedle verify drafts/sample-cloth.json
python3 -m heedle verify drafts/sample-text.json
```

Look at the instruments:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000/web/>.

## Rebuilding generated files

```bash
python3 tools/make_vectors.py          # vectors/*.json, from the Python reference
node tools/make_sample_drafts.mjs      # drafts/*.json, from the JavaScript loom
```

`hosts/text-v1.json` is a trained model. `tools/train_text_host.py` says how it was made. Retraining gives a different model with a different root, so a retrained host gets a new name.

## Commits and pull requests

Sign off every commit (`git commit -s`). Keep a pull request to one change. Say in the description which parts were written or run with AI assistance.

The author of a change does not merge it. See [REVIEWING.md](REVIEWING.md).
