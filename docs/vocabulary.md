# Vocabulary

What each word names in the code. Words that are used for more than one thing are listed after the table.

| Word | What it is | In the code |
|---|---|---|
| host | a small model together with the meaning of its inputs | `hosts.Host`, `web/hosts/*.js` |
| baseline | the weights a session starts from, and the centre of the gate's ball | `host.w0` |
| fell | the last committed weights | `loom.fell` |
| move | a change of the weights, one number per weight | `d` |
| fit probes | the fixed inputs on which the model is linearised | `host.fit_inputs()` |
| shaft | a unit eigenvector of `G = J'J/(n m)` on the fit probes, with its eigenvalue | `Threading.V`, `Threading.lam` |
| shaft pattern | the unit-RMS output pattern a shaft makes | `Threading.U` times `sqrt(n m)` |
| throw | the length of the weight move for one unit of travel along a shaft | `1 / sqrt(lam)` |
| threading | the shafts and their patterns at one fell | `core.Threading`, `thread()` |
| treadle | a named output pattern | `host.treadle_pattern()` |
| tie-up | the coefficients of the treadles on the shafts | `TieUp.alpha` |
| share | the part of a treadle's pattern the shafts in use can carry, to first order, between 0 and 1 | `TieUp.share` |
| treadling | the travels a person sets for one pick. An annotation in a draft | `reading.treadling` |
| heddle | a pinned input: its outputs are held still to first order | `pins` |
| shed | the moves that leave the pinned inputs alone to first order | `core.Shed`, `shedOf()` |
| gate | weft's check on the weights: inside a radius of the baseline, and within budget | `verify_weight_update_safety` |
| claim | "no input changes by more than `tau` since the fell", made by a pick | `claim_id` |
| tension | a forecast of how close a move comes to the gate or the claim | `assess().tension` |
| zone | which gesture a tension asks for: `auto`, `shared` or `knot`. The page also shows `refused` when the gate says no | `zone_of` |
| gesture | how a person committed: `release`, `hold` or `knot` | `gesture` |
| beat | the act of committing a move, and the button for it | `Loom.commit`, `#beat` |
| snap | a move the gate accepted and one input refuted | event kind `snap` |
| unpick | a return to an earlier fell | event kind `unpick` |
| dispute | the host's search for one input that breaks a claim | `host.dispute()` |
| witness | the input a dispute found | `claim.witness` |
| fresh probes | inputs derived from a hash after a candidate is fixed | `host.fresh_inputs()` |
| drawdown | the part of the page that shows what the model does | `#slot`, `views/*.js` |
| futures | what a knob would do at six settings, drawn ahead of the person and discarded when stale | `tendFan()` |
| lamp | the part of the page that says what was checked after a beat | `showLamp()` |
| weave | to sample a line of text from the text host, for the eye | `TextHost.weave` |

## Words used for more than one thing

**Heedle and heddle.** Heedle is the project, in the spelling of the corpus. A heddle is a pinned input.

**Pick.** In a draft, a `pick` is an event: a move that was admitted. In prose a pick is also any move a person beats in, before it is known whether it will be admitted or will snap. "Picks left" on the page counts beats, so a snap spends one.

**Draft.** Here a draft is the hash-chained record of a session, `heedle-draft/1`. A weaver's draft is the notation the page is laid out after. The theory says which it means.

**Loom.** In the code a `Loom` is one session: a host, a fell, pins and a draft. On the page a loom is a host with its view, as in "the text loom". In the origins a loom is the corpus's four-layer design.

**Knot.** A zone and the gesture it asks for have the same name.

**Travel.** How far one knob is turned, in units of first-order output change. `host.travel` is how far the page's knobs can be turned.

**Selvedge and edge.** The theory names three edges of a pick: the proved radius, the predicted change, and the gate. The page draws two bars beside its drawdown and calls them selvedges: the left is the gate, the right is the predicted change. The proved radius is a number beside them.

**Cloth.** The cloth host, and what a session has woven so far on either loom.

## Words from weaving that name nothing here

Warp names nothing in the code. The documents use it only for the corpus's first layer and for the threads of a real loom. Weft is the name of the program this repository belongs to. Shuttle, reed, beam and tapestry name nothing.
