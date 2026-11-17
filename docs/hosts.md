# Hosts

A host is a small model together with the meaning of its inputs. The mathematics in `heedle/core.py` knows a host only as a three-layer network with sparse feature vectors. Everything else is the host's to say.

## The contract

Every host is the network

```
f(x, w) = W3 tanh(W2 tanh(W1 x + b1) + b2) + b3
```

with `F` features, `H` hidden units in each layer and `M` outputs. It keeps this shape so that Theorem 1 of the calibration theory, a proved bound on how far any output can move inside a radius, applies to it as stated.

A host provides:

| What | Why |
|---|---|
| `encode(inputs)` | Turns inputs into sparse feature vectors: per input, a few feature indices and their values |
| `w0` | The baseline weights |
| `fit_inputs()` | The fit probes, on which the model is linearised |
| `held_inputs()` | Inputs kept apart from the fit probes, for studies |
| `known_inputs()`, `elsewhere_inputs()` | A small set of probes a proposer is assumed to know, and inputs away from them, for the hidden-move demonstration |
| `pin_inputs(pins)` | The inputs that a list of integer pins names |
| `fresh_inputs(seed, n)` | Probes derived from a hash, using integer arithmetic only |
| `treadle_pattern(name, inputs, y)` | The output pattern a named change asks for |
| `dispute(w0, w1)` | A search for the one input that changes most, and whether the search was exhaustive |
| `witness_inputs(witness)`, `proved_inputs` in the page | The single input a recorded witness names, or an error if it is outside the domain. The inputs on which the page evaluates the proved bound |
| `descriptor()` | What a draft needs in order to rebuild the baseline |
| `radius`, `tau`, `budget`, `cost` | The gate's radius and the claim's limit, as decimal strings, the number of picks allowed and the cost of one |

Three rules:

1. **Pins are integers.** A pin must hash the same in every language.
2. **Fresh probes come from integers.** A probe may be a fraction, but only one made by dividing a hash word by a power of two.
3. **A dispute says what it read.** It returns how many inputs it evaluated and whether that was all of them.

## cloth/1

A point of the unit square goes in. Three numbers come out and are shown as a colour.

| | |
|---|---|
| Shape | 8 features, 16 hidden units, 3 outputs: 467 weights |
| Features | the two coordinates, and six sines and cosines of them |
| Baseline | drawn from a seed. No training |
| Fit probes | a 10 by 10 grid |
| Pins | cells of a 64 by 64 lattice |
| Treadles | `warm`, `light`, `tilt`, `contrast`, `check` |
| Dispute | reads a 96 by 96 lattice, then climbs from the four best cells. A sampled reading |
| Limits | the gate's radius is 2.0 and the claim's limit is 0.5. They belong to the host: a draft cannot choose others |

Every pixel is an input, and the picture samples the square at 64 or 128 points a side. The domain itself is every point of the square, which no picture and no search reads whole. The cloth is the first host because each idea in the theory can be seen on it. The ball and the ellipsoid, the room between probes, a pin and its neighbourhood are all visible as changes to one picture.

The treadle `check` asks for a fine checkerboard. With sixteen shafts its share is near 0.01. It is kept as a foil: a name the shafts in use cannot carry, shown as such.

The cloth is not trained, and its dispute is a search. Finding nothing means nothing was found among the inputs read.

## text/1

Three characters go in. A score for each possible next character comes out.

| | |
|---|---|
| Alphabet | the 26 letters, the space and the full stop: 28 symbols |
| Input | a context of three characters. There are 21,952 |
| Shape | 84 features, 32 hidden units, 28 outputs: 4,700 weights |
| Features | three one-hot blocks, one per character of the context |
| Outputs | centred scores. A softmax of them is the model's guess at the next character |
| Baseline | trained on this repository's own documents |
| Fit probes | the contexts the training text uses most |
| Pins | contexts |
| Treadles | `vowels`, `spaces`, `stops`, `echo`, `sharp` |
| Dispute | reads every context. Exhaustive |
| Limits | the gate's radius is 2.0 and the claim's limit is 1.0 |

This is about as small as a model of text gets, and it is here for one reason: its domain is finite and small enough to read whole. On the cloth an admitted pick has passed a search. On this host an admitted pick has been checked on every input the model has.

### How it was made

`tools/train_text_host.py` takes nine documents of this repository (the charter, the contributing and reviewing guides, and the six files in `docs/`), removes fenced code, reduces them to the 28 symbols, counts how often each context is followed by each character, and fits the network to those counts. The text it used is kept as `hosts/text-v1.corpus.txt`, with its hash in the host file, so the training text is fixed even when the documents change.

The training text has 44,520 characters and uses 2,499 of the 21,952 contexts. On that text the model spends 1.96 bits per character, against 4.09 for a model that knows only how common each character is. That is a fit to its own training text, not a test. The 48 commonest contexts are the fit probes. The next 512 are held out, and fresh probes are drawn from them in proportion to their use. The eight commonest are the probes a proposer is assumed to know.

The model is named `text-v1`. Retraining gives a different model with a different weight root, so a retrained model gets a new name, and drafts made on the old one still verify against the old file.

### What the treadles ask for

| Treadle | The pattern, on every fit probe | In plain words |
|---|---|---|
| `vowels` | the five vowels up | more vowels |
| `spaces` | the space up | shorter words |
| `stops` | the full stop up | shorter sentences |
| `echo` | the character just written, up | doubled letters |
| `sharp` | the model's current scores, scaled | more of whatever it already prefers |

The first three ask for the same thing in every context. A move of the output biases alone would deliver each of them exactly, but at a length of about 5.3 per unit of travel, outside the gate's radius of 2.0. With sixteen shafts the instrument carries 0.39, 0.30 and 0.16 of them at the baseline. `echo` and `sharp` depend on the context, and sixteen shafts carry less than 0.1 of each.

### What the instrument shows

Each pick weaves one line of text, sampled from the model at the new fell. The line a pending move would weave is drawn below the fell and changes as the knobs turn. Two sets of weights are always compared on the same random draws, so a difference in the text is a difference in the model.

Sampling is for the eye. No check reads a sampled character.

### What it is not

It is a model of three characters of context. It does not know words. Text it weaves looks like the shape of English and means nothing. Its training text is small, and it has learned that text's habits.

Nothing measured on the cloth has been measured on it.

## Adding a host

1. Give it the three-layer shape. If its outputs feed a softmax, centre them.
2. Write `encode` so that features are sparse and indices within one input are distinct.
3. Decide the domain and write the dispute. If the domain can be enumerated, enumerate it and say so.
4. Choose fit probes that cover what the model is used for, and keep held-out inputs apart from them.
5. Write treadles as output patterns. Give each a plain description of what it asks for.
6. Implement it in `heedle/hosts.py` and in `web/hosts/`, and add its vectors with `tools/make_vectors.py`.
7. Add a section here that says what the host is, how its baseline was made, what its dispute reads, and what it is not.

A host whose model is not the three-layer network cannot use the proved bound. It would need its own, or would have to say that it has none.
