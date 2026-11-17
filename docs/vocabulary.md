# Vocabulary

Each word names one thing in the code. Where a word is also used more loosely in prose, the last column says so.

| Word | What it is | In the code | Also |
|---|---|---|---|
| host | a small model together with the meaning of its inputs | `hosts.Host`, `web/hosts/*.js` | |
| baseline | the weights a session starts from, and the centre of the gate's ball | `host.w0` | the warp, in prose |
| fell | the last committed weights | `loom.fell` | |
| move | a change of the weights, one number per weight | `d` | |
| fit probes | the fixed inputs on which the model is linearised | `host.fit_inputs()` | |
| shaft | a unit eigenvector of `G = J'J/(n m)` on the fit probes, with its eigenvalue | `Threading.V`, `Threading.lam` | |
| shaft pattern | the unit output pattern a shaft makes | `Threading.U` | the shaft's picture, on the cloth |
| travel | how far a knob is turned, in units of first-order output change | `t` | |
| throw | the length of the weight move for one unit of travel | `1 / sqrt(lam)` | |
| threading | the shafts and their patterns at one fell | `core.Threading`, `thread()` | |
| treadle | a named output pattern | `host.treadle_pattern()` | |
| tie-up | the coefficients of the treadles on the shafts | `TieUp.alpha` | |
| share | the part of a treadle's pattern the shafts in use can carry, between 0 and 1 | `TieUp.share` | |
| treadling | the travels a person sets for one pick | `reading.treadling` | |
| heddle | a pinned input: its outputs are held still to first order | `pins` | |
| shed | the moves that leave the pinned inputs alone to first order | `core.Shed` | |
| gate | weft's check on the weights: inside a radius of the baseline, and within budget | `verify_weight_update_safety` | the left selvedge, on the page |
| claim | "no input changes by more than `tau`", made by a pick | `claim_id` | the right selvedge shows its forecast |
| tension | a forecast of how close a move comes to the gate or the claim | `assess().tension` | |
| zone | which gesture a tension asks for: `auto`, `shared` or `knot` | `zone_of` | |
| pick | a committed move | event kind `pick` | |
| snap | a move the gate accepted and one input refuted | event kind `snap` | |
| unpick | a return to an earlier fell | event kind `unpick` | |
| dispute | the host's search for one input that breaks a claim | `host.dispute()` | |
| witness | the input a dispute found | `claim.witness` | |
| fresh probes | inputs derived from a hash after a candidate is fixed | `host.fresh_inputs()` | |
| draft | the hash-chained record of a session | `heedle-draft/1` | |
| loom | one session: a host, a fell, pins and a draft | `Loom` in `web/heedle.js` | |

## Two words used twice

**Heedle and heddle.** Heedle is the project, in the spelling of the corpus. A heddle is a pinned input.

**Selvedge.** In prose a selvedge is any edge a pick should stay inside. The theory names three: the proved radius, the predicted change, and the gate. The page draws two bars beside its drawdown and marks the proved radius on one of them.

## Words from weaving that are not used

Warp is used only in prose, for the baseline. Weft is not used for anything here: it is the name of the program this repository belongs to. Shuttle, reed, beam and tapestry name nothing in the code.
