# The draft: a steering session as a record

A draft is the record of one session at a loom. Anyone holding it can rebuild every set of weights the session passed through and check every verdict, without the instrument that made it. This document is the specification of `heedle-draft/1`. The reference verifier is `heedle/draft.py`.

## Design rules

1. **Commit to exact things. Recompute the rest.** An event's identifier covers strings and integers only. Every number with a fraction is a reading, and a verifier recomputes it.
2. **Moves travel as bytes.** A move is carried as float64 values, little-endian. The fell after a pick is the fell before it plus the move, and float64 addition is the same everywhere, so every verifier reaches the same weights.
3. **One input is enough to refute.** A snap carries its witness. Checking it takes two forward passes.
4. **The record is append-only.** Unpicking adds an event. It removes nothing.

## Hashing

All hashes are SHA-256, written as lower-case hexadecimal.

**Canonical JSON** is JSON with keys sorted, no spaces, and only strings, integers and lists of them: the bytes of Python's `json.dumps(obj, sort_keys=True, separators=(",", ":"))`.

**Domain strings** keep hashes of different things apart. Weight roots use `WEFT-CBC-V1:`. Everything else uses `HEEDLE-V1:`.

### Weight root

The root of a set of weights is the weight root of committed-behaviour-change v1, over the six tensors `W1`, `b1`, `W2`, `b2`, `W3`, `b3`:

```
leaf(name)  = SHA-256( "WEFT-CBC-V1:leaf:" + canonical_json({"name", "shape", "dtype": "<f8"}) + ":" + bytes )
root        = SHA-256( "WEFT-CBC-V1:root:" + for each name in sorted order: name + ":" + leaf(name) + ";" )
```

`bytes` are the tensor's float64 values in row-major order, little-endian. A root made here is a valid `w0_root` or `w1_root` of a CBC record.

### Claim

```
claim_id = SHA-256( "HEEDLE-V1:claim:" + canonical_json({
    "schema": "heedle-claim/1", "host": kind, "w0_root", "w1_root", "tau", "pins" }) )
```

`tau` is a decimal string. `pins` are sorted integers. The claim says: no input in the host's domain changes by more than `tau` between `w0_root` and `w1_root`. The change of an input is the root mean square, over its outputs, of the difference.

### Probes

```
probe_seed = SHA-256( "HEEDLE-V1:probes:" + test_set + ":" + w1_root + ":" + claim_id )
word(i, j) = the j-th big-endian 32-bit integer of SHA-256( probe_seed + i as 4 big-endian bytes )
```

The host turns `word(i, j)` into its `i`-th probe. The seed depends on the candidate's root, so the probes cannot be known before the candidate is fixed.

### Event identifier

```
event_id = SHA-256( "HEEDLE-V1:event:" + canonical_json(the event's fields among:
    seq, kind, prev, parent_root, root, move_sha256, pins, gesture, tau) )
```

## The document

```json
{
  "schema": "heedle-draft/1",
  "host": { "kind": "cloth/1", "seed": 7, "w0_root": "..." },
  "gate": { "r_invariant": 2.0, "budget": 24, "cost": 1, "tau": "0.5" },
  "events": [ ... ]
}
```

`host` names the baseline. A `cloth/1` host is rebuilt from its seed. A `text/1` host is named, and its weights are the file `hosts/<name>.json`. In both cases `w0_root` must equal the root of the rebuilt baseline.

`gate` holds the session's limits: the gate's radius around the baseline, the number of picks the budget allows, the cost of one, and the claim's limit.

## Events

Every event has `seq` (from 1), `kind`, `prev` (the previous event's `id`, or `"genesis"`), `parent_root` (the fell before it), `root`, `pins`, `gesture`, `tau` and `id`.

### pick

The gate accepted the move, and the dispute found no input that breaks the claim. The fell becomes `root`.

| Field | Meaning |
|---|---|
| `move_b64` | the move, float64 little-endian, base64 |
| `move_sha256` | the hash of those bytes |
| `gate` | `accepted`, `distance` from the baseline, `remaining_budget`, `reason` |
| `claim` | `id`, `probes` (how many), `realized` (change on the fresh probes), `worst` and `witness` (the dispute's largest change and where), `rows` (inputs read), `exhaustive`, `refuted` |
| `reading` | what the instrument showed before the beat: `length`, `predicted`, `predicted_worst`, `tension`, `zone`, `pin_leak`, `treadling`, `delivered`, `note` |

### snap

The gate accepted the move, and one input breaks the claim. The fell does not move: after a snap the fell is still `parent_root`. `root` is the root the candidate would have had. The budget is spent. The fields are those of a pick, with `claim.refuted` true and `claim.witness` the input that refutes.

### unpick

The person returned to an earlier fell. `root` is that fell, which must have been the fell at some earlier point. The weights return. The budget already spent stays spent.

## Gestures and zones

`gesture` is one of `release`, `hold`, `knot`, in rising order of deliberateness. The **tension** of a pick is the larger of `distance / r_invariant` and the first-order change at the worst fit probe over `tau`. Its **zone** is `auto` below 0.3, `shared` from 0.3, `knot` from 0.7. A pick is well formed only if its gesture is at least as deliberate as its zone.

## What a verifier checks

Starting from the rebuilt baseline, for each event in order:

1. `prev` equals the previous `id`, and `id` equals the event identifier of the event's own fields. **Exact.**
2. `parent_root` equals the current fell's root. **Exact.**
3. For a pick or a snap: `move_sha256` equals the hash of the move's bytes. **Exact.**
4. The gate, run on baseline and candidate with the remaining budget, gives the recorded `accepted` and `reason`. **Exact.** The recorded `distance` matches. **Within a tolerance.**
5. The candidate's weight root equals `root`. **Exact.**
6. The gesture is deliberate enough for the tension the verifier computes. **Skipped within `1e-6` of a threshold.**
7. `claim.id` equals the claim identifier. **Exact.** `claim.realized` matches the change on the probes the verifier derives. **Within a tolerance.**
8. For a snap: the change at `claim.witness` exceeds `tau`, and the witness is in the host's domain. One input, two forward passes.
9. For a pick: the verifier's own dispute finds nothing above `tau`, and its largest change matches `claim.worst`. **Within a tolerance.**
10. For an unpick: `root` was the fell at some earlier point.

A draft passes when none of these fails.

## What a draft cannot show

- **That a person made the gesture.** It records what the instrument saw.
- **That a pick was a good idea.** It records that the pick stayed within its claim.
- **More than the dispute read.** On the cloth, admission is a sampled reading. On the text host the dispute reads the whole domain, in float64.
- **Why.** `reading.treadling` and `reading.note` say what was pressed. They are annotations. The verifier does not check them, and a move need not have come from a treadle at all.

## Versions

A change to anything a hash covers, to the event kinds, or to what a verifier must check, needs a new schema name. Adding a reading does not.

## Use

```bash
python3 -m heedle verify drafts/sample-cloth.json
python3 -m heedle verify drafts/sample-text.json
```

The command prints what it replayed and each difference it found, and exits with status 0 only when there is none.
