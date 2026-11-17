# Charter

Heedle builds the layer where a person steers a model's change by hand, inside bounds that are checked.

## Why this exists

The first design conversation behind ML Weft described a loom in four layers: Warp, Weft, Heedle and Tapestry. The third is where a person and a model meet. It "raises and lowers specific threads of the Warp to allow the Weft to pass through". Weft's runtime builds parts of the other three. Until now nothing built the third.

The design note [The heddle: the layer where a person steers](https://github.com/Moissanite-Labs/weft-loom/blob/main/notes/heddle-steering-instrument.md) in weft-loom proposed to build it, with a first instrument and a first study. This repository is that proposal taken up. The name keeps the corpus's spelling. A real loom has heddles, and inside this repository a heddle is one thing: an input a person has pinned.

## What Heedle is for

One loop, made by hand and checked by a program:

1. **Propose.** The instrument shows what a change would do before it is made.
2. **Steer.** A person sets a named change and how far to take it, in units of behaviour.
3. **Hold.** The person marks what must not move.
4. **Check.** The change commits only through a gate on the weights and a claim about behaviour that one input can refute.
5. **Keep or discard.** A kept change joins a record that anyone can replay. A refuted one leaves the model where it was.

Heedle exists so that this loop can be run by a person at the speed of a hand, on real model internals, with every step open to checking.

## What it covers

- **The mathematics** of knobs marked in units of behaviour, named changes with a share, pins, and tension. One implementation for any host that fits the contract.
- **Hosts**: small models whose inputs a program can enumerate or search. Two exist: a cloth, where every input is a pixel, and a text model, where every input is three characters.
- **Instruments**: pages a person can use, laid out as a weaving draft.
- **The draft**: a hash-chained record of a steering session, and a verifier that replays it without the instrument.
- **Conformance**: the reference in Python and the mirror in JavaScript are held to the same vectors.

## What it does not cover

- **It does not run agents, tools or training.** It changes the weights of a small model along directions a person chose.
- **It is not the Weft runtime.** It uses weft's gate unchanged and adds nothing to it.
- **It does not hold experiment records.** Studies of Heedle belong in [weft-loom](https://github.com/Moissanite-Labs/weft-loom), preregistered there and reviewed there. This repository links them.
- **It makes no claim about large models.** Every host here is a toy chosen so that its whole domain can be read. Whether the loop survives a model whose inputs cannot be enumerated is an open question, listed in the roadmap.
- **It does not prove a person was present.** A draft records the gesture the instrument saw.

## Rules

These follow the Siliconcordat and weft's rules for evidence.

1. **A stated rule is not a checked rule.** A document that says the instrument enforces something links the test that checks it, or says it is not checked.
2. **A passing test is not a proof.** The theory marks what follows from the definitions, what a test checks on a toy host, and what is a reading.
3. **A forecast never refuses.** Tension is a first-order prediction. Only the gate refuses, and only a refuted claim undoes a pick.
4. **The person decides.** The model may propose a change, a name or a pin. Nothing commits without the person's gesture, and the closer a change comes to its limits the more deliberate the gesture must be.
5. **Learning a preference does not grant permission.** An instrument may remember how a person likes to steer. Each pick still needs the gesture.
6. **Every session can be replayed.** If a verifier cannot rebuild a weight root from the draft alone, that is a bug.
7. **Say what is toy.** No document here describes a result on a toy host as a law.

## Relations

| Project | What Heedle takes from it | What Heedle gives back |
|---|---|---|
| weft | The gate, vendored unchanged and pinned by hash | A place where a person can see the gate, the claim and the proved bound side by side |
| weft-loom | The calibration theory, the committed-behaviour-change design, the note and the first record | Hosts and an instrument for further records |
| ethos | The rule that a preference is not a permission | A form a steering preference could travel in: a tie-up and a treadling |
| exo | The idea of approvals and a visible trace | Approvals that scale with a forecast, and a draft as the trace |
| synmerit | The idea of signed records of contribution | Per-pick records that two people could divide exactly, to first order, along orthogonal shafts |

The right column lists what could be given. None of it is wired up.

## Owner and review

Owner: `abu-aska`, who created this repository. Change this line when someone else takes it on.

Changes are reviewed as in [REVIEWING.md](REVIEWING.md). The author of a change does not merge it.

## What counts as done, for now

The promotion rule of the Siliconcordat's ideas register asks for an owner, a charter, and a first study or runnable thing. This file is the charter. The instruments run. The first study is the record `heddle-knob-range-2026-11-17` in weft-loom, made on the prototype this repository grew from.

## Disclosure

This charter and the first version of this repository were written by an AI agent session (Claude) at the request of the repository owner.
