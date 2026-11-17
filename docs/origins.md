# Origins

Heedle has two sources. Neither knew of the other.

## A loom in a conversation

ML Weft began as a long design conversation, kept word for word in [weft-corpus](https://github.com/Moissanite-Labs/weft-corpus). Early in it (`weft/browser-sources/thread1-splicers-genesis-90133B.md`, lines 254 to 259) a loom is laid out in four layers. The third:

> The **Heedle** is the mechanism that raises and lowers specific threads of the Warp to allow the Weft to pass through. This is where human intuition and machine calculation achieve perfect synthesis.

The same conversation asks for an interface through which a person feels a system's strain and adjusts it by hand (lines 341 to 345), and for a governor that "uses the tension of the loom as a real-time dial for autonomy" (lines 431 to 456). Its dial has three zones. Below a tension of 0.3 the machine weaves alone. From 0.3 the person is drawn in. From 0.7 the weave waits for the person.

Weft's own reading of the corpus classes all of this as vision language: words for what a design aims at, not a description of code. The glossary in weft keeps such words and writes beside each one what the code does today. It has no entry for this layer, because nothing built it.

Heedle keeps three things from the conversation and treats them as proposals to test:

- the layer itself, and its name;
- the three zones and their thresholds, which nothing derives;
- the idea that a boundary should cost more to approach instead of being a wall. The conversation's price was compute. Heedle's is the person's deliberateness: a longer hold.

## A cockpit for a story

[Spaceship IDE](https://github.com/chuanqisun/spaceship) is a design concept by Chuanqi Sun of the MIT Media Lab, released under the MIT licence. Heedle read it at commit `4ddd4d1061aa5f22113d5dd03462b3b3e5f506a6`. Its README states one conjecture, that language models and decision models will soon answer fast enough to interact with sixty times a second, and asks what a person could then do. It shows a story being written while the reader steers it, with four ideas:

| Idea | What the reader does |
|---|---|
| Focus monitor | sees which earlier phrases the next words depend on |
| Attention lock | keeps, moves, removes or adds boxes on phrases, and later text is built on them |
| Future vector steering | turns knobs for the text to come, each an axis with a position and a strength |
| Realtime re-generation | drags a knob back through the text, and what lies ahead is written again |

Three things in its source shaped Heedle more than the four names did.

**It checks its own steering.** A second model is asked of each new sentence whether the sentence delivers what the control asked for. A control is treated as a claim and a sentence as evidence. That is the stance of this lab's records, reached from the side of the interface.

**The model proposes and the person disposes.** The model suggests what matters. The reader decides what stays.

**The work is in what to throw away.** Its README says the hard part was not generating ahead but deciding when what was generated is stale. Heedle's instruments count the futures they weave ahead and the ones they discard.

What Spaceship does not have is what this lab has: the model's own internals, a bound, and a record. In its live mode a fast text model is prompted to pick phrases and axes, and a knob reaches the model as words in a prompt. By default the page runs simulated sources. Its README frames each idea as a question.

Heedle uses no code from Spaceship. It takes the four ideas and gives each a counterpart on real internals:

| Spaceship's idea | Heedle's counterpart | How they differ |
|---|---|---|
| Focus monitor | The instrument dims whatever the pending move leaves unchanged | Heedle shows what a change touches. Spaceship shows what the next words depend on |
| Attention lock | A heddle: a pinned input | A locked phrase leads the next text. A pin forbids change |
| Future vector steering | Shafts and treadles, in units of behaviour, each with a share | Heedle's directions are eigenvectors of a matrix computed from the model |
| Realtime re-generation | Futures woven ahead of the knob, and unpicking | Heedle keeps what was unpicked in the record |

If you build on Spaceship's ideas, cite Spaceship. Its repository has a `CITATION.cff`.

## The note and the first record

The connection was noticed by the owner of this repository on 17 November 2026. The design note in weft-loom, `notes/heddle-steering-instrument.md`, worked it out, with a prototype and a preregistered record, `experiments/heddle-knob-range-2026-11-17`. A second agent session fact-checked the note, and a member of the lab reviewed and approved the pull request, including a run of the tests on a second machine.

This repository starts from that prototype. Three things are new here. The mathematics no longer knows what a host's inputs mean, so one implementation serves any host. There is a second host, a small text model, whose dispute reads its whole domain. And the draft has a written specification.

## Weaving

The vocabulary is a naming choice. It was made because the first-order picture of a pick has the shape of a weaver's draft: a product of a threading, a tie-up and a treadling. [vocabulary.md](vocabulary.md) lists what each word names in the code. The words prove nothing.

One word needs a sentence. The repository is named Heedle, in the corpus's spelling. The loom part is a heddle. Here a heddle is a pinned input, and Heedle is the project.
