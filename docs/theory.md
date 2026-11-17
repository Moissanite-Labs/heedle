# Theory: a knob is a claim with a handle

This document says what the instrument computes and why. It builds on two documents in weft-loom: the calibration theory (`experiments/gate-metric-calibration-2026-11-15/THEORY.md`) and the committed-behaviour-change design (`experiments/committed-behaviour-change-2026-11-15/DESIGN.md`). Those judge a move after it is made. This one uses the same objects to make a move.

Marks:

- **[P]** follows from the definitions by linear algebra, to first order where the sentence says so.
- **[T]** checked by a test in this repository, on its toy hosts. The test is named.
- **[K]** a known result used as stated.
- **[I]** a reading or a proposal.

A passing test is not a proof, so [T] is never used where [P] is meant.

## 1. Setting

A host computes `f(x, w)` from an input `x` and a weight vector `w` of length `P`. Every host here is the three-layer network

```
f(x, w) = W3 g(W2 g(W1 x + b1) + b2) + b3,      g = tanh
```

with a fixed map from its inputs to feature vectors `x`. A host may centre its outputs.

The **fell** is the last committed weight vector. A **move** is a vector `d`, and the candidate is `fell + d`.

On a set `X` of `n` inputs with `m` outputs each, `J` is the matrix of derivatives of the outputs with respect to the weights, and `G = J'J / (n m)`. [K] For a small move the output change is `J d`, so the root-mean-square change on `X` is `sqrt(d'Gd)` to first order.

The **fit probes** are the fixed set of inputs on which `J` is taken. The **change** of one input is the root mean square, over its outputs, of the difference between candidate and fell.

## 2. Shafts: knobs in units of behaviour

Take the eigenvectors `v_i` of `G` with eigenvalues `lam_i`, largest first. They are the **shafts**.

[P] To first order, a move of `t / sqrt(lam_i)` along `v_i` changes the outputs on the fit probes by `t` in root mean square, whichever shaft it is. Proof: `d'Gd = t² v_i'G v_i / lam_i = t²`.

So a knob on a shaft can be marked in **travel**, a unit of behaviour, and one unit means the same on every knob. A stiff shaft has a short throw in weight space and a slack one a long throw.

[P] Each shaft has a pattern. With `J v_i = s_i u_i` and `u_i` of unit length, the output change of one unit of travel is `u_i sqrt(n m)`, a pattern of unit root mean square.

[T] `tests/test_core.py::test_travel_is_measured_in_behaviour` checks the first statement on both hosts.

## 3. Treadles, the tie-up and the share

A **treadle** is a named output pattern `c` on the fit probes, of unit root mean square. Its coefficient on shaft `i` is `alpha_i = u_i'c / sqrt(n m)`. With travel `t`, tied to the first `K` shafts, its move is

```
d = t * sum over i < K of alpha_i v_i / sqrt(lam_i)
```

[P] To first order the output change is `t` times the projection of `c` onto the span of the `K` shaft patterns. Its gain along `c`, `<J d, c> / <c, c>`, is `t * rho` with

```
rho = sum over i < K of alpha_i²
```

Call `rho` the **share** of the treadle. It lies between 0 and 1. The part of the request that is not delivered has root mean square `t * sqrt(1 - rho)`.

[P] The share never falls when shafts are added, because each added term is a square.

[I] So a label comes with a number that says how much of it is true, to first order, at the current weights, for the shafts in use. A low share does not mean the model cannot make the change. It means these shafts cannot carry it.

The matrix of the `alpha`, shafts by treadles, is the **tie-up**.

[T] `test_tie_up_delivers_its_share` and `test_more_shafts_never_carry_less`.

## 4. A treadling is a weaving draft

Write `Theta` for the shaft patterns, one column per shaft, `T` for the tie-up, and `k` for the vector of travels a person sets. [P] Then to first order

```
output change = Theta T k
```

Proof: `d = V L^(-1/2) T k` with `L` the diagonal matrix of the `lam_i`, and `J V = Theta L^(1/2)`.

[K] A weaver's draft has the same shape. The threading says which warp thread hangs on which shaft, the tie-up says which shafts each treadle lifts, the treadling says which treadle is pressed for each pick, and the drawdown is their product taken with "and" and "or".

[I] The likeness is exact in form for one pick, to first order. A loom's threading is fixed. This one changes after every pick, because `G` does. The likeness is still useful: it names three things with three different owners. The threading is computed from the model. The tie-up belongs to whoever chose the names. The treadling is the person's.

[T] `test_a_treadling_is_a_weaving_draft` checks the identity to `1e-9` on both hosts.

## 5. Heddles and the shed

A **heddle** is a pinned input: its outputs are to be held still. Let `J_pin` be the Jacobian on the pinned inputs. The **shed** is the null space of `J_pin`.

[P] A move in the shed leaves the pinned outputs unchanged to first order.

The instrument threads the loom around the pins: its shafts are the eigenvectors of `G` restricted to the shed. [P] Each such shaft lies in the shed. [P] Restricting to the shed cannot raise an eigenvalue, so pins never add reach.

[T] What reaches a pin is second order: halving a move quarters the leak (`test_what_reaches_the_pins_is_second_order`).

[I] Section 6 of the calibration theory notes that the same subspace is read two ways in Weft's records: as the safe channel, and as the hiding place. "The difference between the two readings is who chose the direction and what they knew." A heddle settles who chose: the person, by hand, where it can be seen. The same construction, used by a proposer who knows which probes will be read, hides a change. `core.hidden_move` builds one, and the instrument has a button that shows it.

## 6. Three edges

A pick has three edges.

1. **Proved.** [K] Theorem 1 of the calibration theory. Its proof needs only a bound on the units' slope, here 1 for tanh. Inside radius `R` no input's output moves by more than `R c(x) + R² q(x) + R³ k(x)`, with `c`, `q` and `k` computed at the fell. Centring the outputs is a projection, so the bound holds for centred outputs. [T] No move tried in `test_the_proved_bound_is_never_exceeded` exceeds it.
2. **Predicted.** The first-order change at the worst fit probe, against the claim's limit `tau`.
3. **Permitted.** Weft's gate: `‖w − w₀‖ ≤ R` against the session's baseline, and `budget ≥ cost > 0`.

[I] Along a host's shafts the three usually nest, the proved edge innermost. In general they do not. The set of moves with a small predicted change is unbounded along directions the fit probes cannot see, and so runs outside the gate's ball. This is the calibration record's point: a ball in weight space is the wrong shape for a bound on behaviour.

## 7. Tension and the gesture

The **tension** of a move is the larger of two numbers: its distance from the baseline over the gate's radius, and its predicted change at the worst fit probe over `tau`.

Tension sets how deliberate the person's gesture must be. The thresholds are the corpus's:

| Tension | Gesture |
|---|---|
| below 0.3 | a tap |
| 0.3 to 0.7 | press and hold |
| 0.7 and up | a longer hold |

[I] Tension is a forecast. It never refuses a move. The gate refuses, and a refuted claim undoes a pick. A forecast that refused would turn a first-order estimate into a rule.

[I] Nothing derives 0.3 and 0.7. In the first record no pick below tension 0.7 was refuted, and the lowest refuted pick was at 0.88, so that record does not single out 0.7.

## 8. A pick is a committed behaviour change

When a person beats a move in:

1. The gate decides on the weights. A refusal records nothing and spends nothing.
2. The candidate gets a weight root. [T] It is the root of committed-behaviour-change v1, byte for byte (`test_weight_root_is_the_cbc_weight_root`, which reimplements that scheme independently).
3. The pick makes a **claim**: no input in the host's domain changes by more than `tau`.
4. Probes are derived from the hash of the root and the claim, so after the candidate is fixed. They measure what the pick did. They do not decide it.
5. The host's **dispute** looks for one input that breaks the claim. If it finds one, the pick **snaps**: the fell does not move, the input is kept as the reason, and the budget is spent. If it finds none, the pick is admitted.

A snap is verified from its one witness: the input, two forward passes, a comparison.

## 9. When the dispute can be complete

How much an admission says depends on the host's domain.

[I] On the cloth the domain is a square. The dispute reads a lattice and climbs from the best cells. Finding nothing is a sampled reading: a larger search could find more.

[P] On a host whose domain is finite and is enumerated, finding nothing is complete: the largest change among all inputs is the largest there is. The text host's domain is every context of three characters, 21,952 in all, and its dispute reads every one. An admitted pick there has been checked on the whole domain.

[I] Two cautions. The check is in float64, so "complete" means complete up to rounding. And a complete check of a claim says nothing about whether the claim was the right one to make: a model can stay within `tau` everywhere and still change in a way a person cares about.

[T] `test_text_dispute_reads_the_whole_domain`.

## 10. The draft

Every pick, snap and unpick is an event in a hash chain. An event commits to exact things only: its place in the chain, the roots before and after, the hash of the move's bytes, the pins, the gesture and `tau`. The move travels as float64 bytes.

[P] The fell after a pick is the fell before it plus the recorded move, added in float64. Addition of two float64 numbers is the same on every machine, so a verifier that starts from the same baseline reaches the same bytes and the same root.

Readings in an event are not trusted. A verifier recomputes them.

[I] A draft records the gesture the instrument saw. Nothing proves a person made it.

[T] `tests/test_draft.py` replays drafts made by the JavaScript loom and catches a changed move, a reordered event, a snap whose witness does not refute, and a gesture too light for its tension.

## 11. What has been measured

One record exists: `experiments/heddle-knob-range-2026-11-17` in weft-loom, made on the cloth host of the prototype this repository grew from. Seventeen predictions were fixed before the run, on 12 random and 12 fitted hosts. Sixteen were confirmed in each arm. Two were not, and the record says which.

In short, on those hosts: a treadle's label held within a factor 1.1 up to a travel of 0.6 on 23 of 24 hosts. No pick with tension below 0.7 was refuted and all but one with tension 1.0 or more were. Sixteen pins closed a tenth of the directions and left between 0.28 and 0.60 of the named reach. A move hidden from sixteen known probes changed the rest of the cloth between 4.6 and 13 times more than those probes showed.

Nothing has been measured on the text host. Nothing has been measured with a person.

## Not covered

- No bound is proved for any host that is not the three-layer network above.
- The first-order picture is a picture. How far it holds is a measurement, made so far on one host shape.
- The hidden move is a first-order construction. A stronger adversary would find more room.
- The theory says nothing about whether a person steers better with any of this.
