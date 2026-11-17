# Roadmap

What could grow from here. Each entry is an idea. Each says what exists and what it would take to find out whether the idea is a good one. The first two also say what would count against them. Nothing here is adopted.

The entries began as the twelve seeds of the design note in weft-loom. Where this repository has taken a step, the entry says so.

## Hosts

### 1. A text host a person would recognise

**Exists.** `text/1`: three characters in, the next one out, 4,700 weights. Its dispute reads all 21,952 contexts, so an admitted pick has been checked on the whole domain.

**Next.** A model with a longer memory. Past a handful of characters the domain cannot be enumerated, and the dispute becomes a search again. The question is what that search should be for text. Candidates: every context that occurs in a held corpus; contexts found by following the model's own samples; a proposer and a disputer drawn from different sources.

**Against it.** If no search finds the changes a person would call harmful on a model whose domain cannot be read whole, the loop does not scale by this route.

### 2. Directions read from a real language model

**Exists.** Nothing.

**Next.** On one small open model: find directions in weights or activations for a few named changes, compute each one's share on fit text, and measure its gain on held-out text. Spaceship's check, a second model asked whether a sentence delivers a control, can be run beside it on the same sentences.

**Against it.** If no named direction keeps its label on held-out text, shares on a real model are not worth showing.

### 3. A fell for Plasm

**Exists.** Nothing. Weft's Plasm cycle proposes two candidates per generation and keeps at most one.

**Next.** Export one recorded Plasm run as a draft, with the candidate that was not chosen as a discarded future, and replay it. This needs a host for Plasm's weight matrix and no new mathematics.

## The instrument

### 4. Names the model proposes

**Exists.** Each host has five treadles, written by hand.

**Next.** Let a model propose patterns for the shafts it sees. Each proposal arrives with its share. The person accepts a name or does not. Test: are proposed names carried better than the hand-written ones, and do people choose them?

### 5. Two weavers on one cloth

**Exists.** Nothing.

**Next.** Shafts are orthogonal. If two people hold disjoint shafts, each pick's first-order change divides between them exactly. A draft could carry both gestures, and Synmerit's signed records could attach to it. Test: with two scripted weavers, how fast does the division go wrong as travel grows?

### 6. Sound and touch

**Exists.** Nothing. The corpus imagined feeling a system's strain. Spaceship's word tick grows softer the faster its main knob is dragged.

**Next.** Tension as pitch. A snap as a snap. Test: with eyes on the drawdown only, does a person stop before a pick would be refuted?

### 7. The room between probes, shown

**Exists.** `core.hidden_move`, and a button that shows one move built to hide from probes it knows.

**Next.** A number beside the probes: how much change could hide from these. The first record found that it falls as the probes multiply.

### 8. Autopilot inside the first zone

**Exists.** Zones, and a gesture that grows with tension.

**Next.** Below tension 0.3 the corpus lets the machine weave alone. Let a proposer make small picks there with the person at rest, every pick in the draft. Test: how often is such a pick unpicked later?

## The record

### 9. Signed drafts

**Exists.** A hash chain that starts from the draft's header. No signatures, so anyone can write a draft that verifies by making picks that obey the rules.

**Next.** A signature per event, by the person's key and the instrument's. Then a draft can say who stands behind each gesture, though still not that a hand made it.

### 10. Snaps as closures

**Exists.** Snaps, each with one witness.

**Next.** weft-loom's closure notes close a path only on a certificate, and admit a counterexample as one. A snap is a counterexample. A register of snaps is a map of where a model cannot be taken under a given claim.

### 11. A steering preference that travels

**Exists.** Nothing.

**Next.** A person's tie-up and habitual travels, written in a form a different host could re-thread. This is where Ethos would attach. The rule holds as written: learning a preference does not grant permission.

## Studies

Studies belong in weft-loom, preregistered. These are the ones this repository makes possible.

### 12. Does the forecast hold on text?

The first record found the forecast close on the cloth. On `text/1` the check is exhaustive, so the comparison is exact: every pick forecast safe, against every context there is.

### 13. Where sixty frames a second ends

Frame time against host size, with the linear algebra moved to a worker or to WebGPU. Spaceship's conjecture is about the future. For small models it can be measured now.

### 14. People

Does a person reach a target faster, or with fewer refuted picks, with treadles than with shafts, and with either than with raw weights? Everything measured so far is about the instrument. This is the question the layer exists for. It needs people, their consent, and a task.

## Open questions

1. Shafts change after every pick. How should a knob keep its meaning along a session? The instrument re-threads at each fell and re-ties the same named patterns.
2. A pin holds its neighbourhood back, by an amount that varied more than a pilot suggested. Can the instrument show the extent of a pin before the pick?
3. Is 0.7 the right threshold for a longer hold? Nothing derives it, and the first record does not single it out.
4. Should a refuted pick spend the budget? Here it does, so that probing the limit is not free.
5. On the cloth the dispute starts from a fixed lattice that a proposer could know. Should it start from the fresh probes instead, as the committed-behaviour-change design's search does?
6. What is the right claim? "No input changes by more than `tau`" is the one the records use. A person steering wants change, in one place and not in others.
7. Which of this survives a model whose weights are too many to hold a Jacobian for?
