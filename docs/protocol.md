# The draft: a steering session as a record

A draft is the record of one session at a loom. Anyone holding it, and the host it names, can rebuild every set of weights the session passed through and check every verdict, without the instrument that made it. This document is the specification of `heedle-draft/1`. The reference verifier is `heedle/draft.py`. Each rule below has a label, the verifier reports differences by label, and `tests/test_draft.py` holds a tampered draft for every label.

A draft that breaks a rule here and still passes the verifier is a security bug.

## Design rules

1. **Commit to exact things.** Every hash covers strings and integers only. A hashed object that holds anything else is refused.
2. **The limits belong to the host.** A draft states the session's limits, and they must be the host's. A draft cannot choose its own.
3. **Moves travel as bytes.** A move is carried as float64 values, little-endian. The fell after a pick is the fell before it plus the move, and float64 addition is the same on every machine, so every verifier reaches the same weights.
4. **One input is enough to refute.** A snap carries its witness. Checking it takes two forward passes.
5. **The record is append-only.** Unpicking adds an event. It removes nothing.
6. **Readings are annotations.** What the instrument showed a person before a beat is kept under `reading`. A verifier does not check it.

## Hashing

All hashes are SHA-256. Where a hash is written into a draft, or enters another hash as text, it is lower-case hexadecimal. The probe seed is the one hash used as raw bytes.

**Canonical JSON** is JSON with keys sorted, no spaces, and only strings, integers and lists of them: the bytes of Python's `json.dumps(obj, sort_keys=True, separators=(",", ":"))`. Numbers with a fraction are never hashed, so limits are written as decimal strings.

**Domain strings** keep hashes of different things apart. Weight roots use `WEFT-CBC-V1:`. Everything else uses `HEEDLE-V1:`.

### Weight root

The root of a set of weights is the weight root of committed-behaviour-change v1, over the six tensors `W1`, `b1`, `W2`, `b2`, `W3`, `b3`:

```
leaf(name)  = hex SHA-256( "WEFT-CBC-V1:leaf:" + canonical_json({"name", "shape", "dtype": "<f8"}) + ":" + bytes )
root        = hex SHA-256( "WEFT-CBC-V1:root:" + for each name in sorted order: name + ":" + leaf(name) + ";" )
```

`bytes` are the tensor's float64 values in row-major order, little-endian. A root made here is a valid `w0_root` or `w1_root` of a CBC record.

### Genesis

```
genesis_id = hex SHA-256( "HEEDLE-V1:draft:" + canonical_json({"schema": "heedle-draft/1", "host", "gate"}) )
```

The first event chains from it, so every event is bound to the host and to the limits.

### Claim

```
claim_id = hex SHA-256( "HEEDLE-V1:claim:" + canonical_json({
    "schema": "heedle-claim/1", "host": kind, "w0_root": parent_root, "w1_root": root, "tau", "pins" }) )
```

`w0_root` is the fell the event starts from, which is the event's `parent_root`. It is not the session's baseline. The claim says: no input in the host's domain changes by more than `tau` between those two sets of weights. The change of an input is the root mean square, over its outputs, of the difference.

A claim is made pick by pick. Nothing in a draft limits how far several picks together take a model from its baseline, except the gate.

### Probes

```
probe_seed = SHA-256( "HEEDLE-V1:probes:admission:" + root + ":" + claim_id )        (32 raw bytes)
word(i, j) = the j-th big-endian 32-bit integer of SHA-256( probe_seed + i as 4 big-endian bytes )
```

The host turns `word(i, j)` into its `i`-th probe. The seed depends on the candidate's root, so the probes of a candidate are fixed only once the candidate is. A proposer can still compute the probes of any candidate it considers. Probes measure a pick. They never decide one.

### Event identifier

```
event_id = hex SHA-256( "HEEDLE-V1:event:" + canonical_json(the event's fields among:
    seq, kind, prev, parent_root, root, move_sha256, pins, gesture, tau) )
```

## The document

```json
{
  "schema": "heedle-draft/1",
  "host": { "kind": "cloth/1", "seed": 7, "w0_root": "..." },
  "gate": { "r_invariant": "2.0", "budget": 24, "cost": 1, "tau": "0.5" },
  "events": [ ... ]
}
```

`host` names the baseline. A `cloth/1` host is rebuilt from its integer seed. A `text/1` host has a plain name, and its weights are the file `hosts/<name>.json`.

`gate` holds the session's limits: the gate's radius around the baseline and the claim's limit, both as decimal strings, the number of picks the budget allows, and the cost of one.

| Rule | What must hold |
|---|---|
| **H1** | `schema` is `heedle-draft/1` |
| **H2** | `host` names a host: a known kind, an integer seed or a plain name with a file |
| **H3** | `host` is exactly that host's descriptor, including the root of its baseline |
| **H4** | `gate` is exactly that host's limits |

## Events

Every event has `seq`, `kind`, `prev`, `parent_root`, `root`, `pins`, `gesture`, `tau` and `id`.

| Rule | What must hold |
|---|---|
| **E1** | `seq` is the event's position, counted from 1 |
| **E2** | `prev` is the previous event's `id`, or `genesis_id` for the first |
| **E3** | `id` is the event identifier of the event's own fields |
| **E4** | `parent_root` is the root of the fell |
| **E5** | `kind` is `pick`, `snap` or `unpick`. `gesture` is `release`, `hold` or `knot`. `tau` is the session's. `pins` are sorted, distinct integers of the host's domain |

### pick and snap

A **pick** is a move the gate accepted and no input refuted. The fell becomes `root`.

A **snap** is a move the gate accepted and one input refuted. The fell does not move: after a snap it is still `parent_root`, and `root` is the root the candidate would have had. The budget is spent.

| Field | Meaning |
|---|---|
| `move_b64` | the move, float64 little-endian, base64 |
| `move_sha256` | the hash of those bytes |
| `gate` | `accepted`, `distance` from the baseline, `remaining_budget`, `reason` |
| `claim` | `id`, `probes` (how many), `realized` (change on the fresh probes), `worst` and `witness` (the dispute's largest change and where), `rows` (inputs read), `exhaustive`, `refuted` |
| `reading` | annotations: what the instrument showed before the beat |

| Rule | What must hold |
|---|---|
| **M1** | The move is one float64 per weight, and `move_sha256` is the hash of its bytes |
| **G1** | The gate, run on the baseline and the candidate with the budget left, accepts, and gives the recorded `accepted` and `reason`. A move the gate refuses is never an event |
| **G2** | `gate.distance` and `gate.remaining_budget` are the gate's |
| **R1** | `root` is the candidate's weight root |
| **P1** | If the event has pins, the first-order change of the move at every pinned input is at most `1e-6` |
| **T1** | The gesture is deliberate enough for the event's tension |
| **C1** | `claim.id` is the claim identifier |
| **C2** | `claim.probes` is the host's number of fresh probes, and `claim.realized` is the change on them |
| **C3** | `claim.exhaustive` and `claim.rows` say what the host's dispute reads |
| **C4** | `claim.witness` is an input of the host's domain, and its change is `claim.worst` |
| **S1** | For a snap: the change at the witness exceeds `tau`, and `claim.refuted` is true |
| **K1** | For a pick: the verifier's own dispute finds no change above `tau`, and `claim.refuted` is false |
| **K2** | For a pick: the largest change the verifier's dispute finds is `claim.worst` |

**Tension** is the larger of `distance / r_invariant` and the first-order change at the worst fit probe over `tau`, both taken at the fell the event starts from. Its **zone** is `auto` below 0.3, `shared` from 0.3, `knot` from 0.7. A gesture is deliberate enough when it is at least as far along `release`, `hold`, `knot` as the zone is along `auto`, `shared`, `knot`. Within `1e-6` of a threshold (`0.3` or `0.7`), T1 accepts the immediately adjacent lighter gesture (`release` near `0.3`, `hold` near `0.7`), not a gesture two zones lighter.

**Tolerances.** G2's distance, C2's change and C4 are compared within `1e-6 * max(1, value)`. K2 uses the same on a host whose dispute is exhaustive, and `1e-3 * max(1, value)` on a host whose dispute is a search, because the last steps of a search can differ between machines.

### unpick

The person returned to an earlier fell. The weights return. The budget already spent stays spent. An unpick has no move, gate or claim.

| Rule | What must hold |
|---|---|
| **U1** | `root` was the root of the fell at some earlier point of the draft |

## Malformed drafts

A draft with a missing field, a field of the wrong type, a move of the wrong length or a witness that is not an input does not verify. The verifier reports it as malformed. It does not stop with an error.

## What a draft cannot show

- **That a person made the gesture.** It records what the instrument saw.
- **That a pick was a good idea.** It records that the pick stayed within its claim.
- **More than the dispute read.** On the cloth, admission is a sampled reading. On the text host the dispute reads the whole domain, in float64.
- **Why.** Everything under `reading` is an annotation: the tension the page showed, the treadles pressed, a note. The verifier does not check it. A move need not have come from a treadle at all.
- **Who.** There are no signatures. Anyone can write a draft that verifies, by making picks that obey the rules.

## Versions

A change to anything a hash covers, to the event kinds, or to what a verifier must check, needs a new schema name. Adding an annotation does not.

## Use

```bash
python3 -m heedle verify drafts/sample-cloth.json
python3 -m heedle verify drafts/sample-text.json
```

The command prints what it replayed and each difference it found, by rule, and exits with status 0 only when there is none.
