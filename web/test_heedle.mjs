// Tests for Heedle's browser core, on both hosts.   node --test web/test_heedle.mjs
// vectors/*.json are written by the Python reference (tools/make_vectors.py); these tests
// check that web/ reproduces them, and that a loom keeps its own rules.
import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import test from "node:test";
import * as hd from "./heedle.js";
import { makeClothHost } from "./hosts/cloth.js";
import { makeTextHost } from "./hosts/text.js";

const read = (path) => JSON.parse(fs.readFileSync(new URL(path, import.meta.url), "utf8"));
const SPECIAL = { Inf: Infinity, "-Inf": -Infinity, NaN: NaN };
const decode = (x) => (typeof x === "string" && x in SPECIAL ? SPECIAL[x] : x);
const close = (got, want, { rtol = 0, atol = 0 }, label = "") =>
  assert.ok(Math.abs(got - want) <= atol + rtol * Math.abs(want), `${label} ${got} vs ${want}`);

const textData = read("../hosts/text-v1.json");
const benches = {
  cloth: { host: makeClothHost(7), vec: read("../vectors/cloth.json"), inputs: (v) => Float64Array.from(v.inputs.flat()), options: {} },
  text: { host: makeTextHost(textData), vec: read("../vectors/text.json"), inputs: (v) => Int32Array.from(v.inputs), options: { iterations: 60, oversample: 16 } },
};
for (const b of Object.values(benches)) {
  b.w = b.host.w0;
  b.move = Float64Array.from({ length: b.host.P }, (_, i) => ((((i * 37) % 101) - 50) / 50.0) * b.vec.move_scale);
  b.fitInputs = b.host.fitInputs();
  b.fit = b.host.encode(b.fitInputs);
  b.some = b.host.encode(b.inputs(b.vec));
}
const each = (name, body) => {
  for (const [key, b] of Object.entries(benches)) test(`${key}: ${name}`, () => body(b, key));
};

each("the weights and every hash match the reference bit for bit", (b) => {
  const { host, vec, w, move } = b;
  const draw = hd.mulberry32(12345);
  assert.deepEqual(Array.from({ length: 8 }, () => draw()), vec.exact.mulberry32_12345);
  assert.equal(host.P, vec.exact.weights);
  assert.deepEqual(Array.from(w.slice(0, 4)), vec.exact.weights_head);
  assert.equal(hd.weightRoot(host, w), vec.exact.w0_root);
  assert.deepEqual(host.descriptor(), vec.host);
  const w1 = hd.addScaled(w, move, 1.0);
  assert.equal(hd.weightRoot(host, w1), vec.exact.w1_root);
  const cid = hd.claimId(host.kind, vec.exact.w0_root, vec.exact.w1_root, host.tau, vec.pins);
  assert.equal(cid, vec.exact.claim_id);
  assert.deepEqual(Array.from(host.freshInputs(hd.probeSeed(vec.exact.w1_root, cid), 8)), vec.exact.fresh_inputs.flat());
  const ev = { seq: 1, kind: "pick", prev: "genesis", parent_root: vec.exact.w0_root, root: vec.exact.w1_root, move_sha256: "00".repeat(32), pins: vec.pins, gesture: "hold", tau: host.tau };
  assert.equal(hd.eventId(ev), vec.exact.event_id);
});

test("the built-in SHA-256 agrees with Node's", () => {
  for (const n of [0, 1, 55, 56, 63, 64, 65, 3736, 37600]) {
    const bytes = Uint8Array.from({ length: n }, (_, i) => (i * 131 + 7) & 255);
    assert.equal(hd.toHex(hd.sha256(bytes)), crypto.createHash("sha256").update(bytes).digest("hex"));
  }
});

each("forward pass and both derivatives match the reference", (b) => {
  const { host, vec, w, move, some } = b;
  const n = vec.numeric;
  const y = hd.forward(host, w, some);
  n.forward.values.forEach((want, i) => close(y[i], want, n.forward, "forward"));
  const Jd = hd.jvp(host, w, some, move);
  n.jvp.values.forEach((want, i) => close(Jd[i], want, n.jvp, "J d"));
  const r = Float64Array.from({ length: some.n * host.M }, (_, j) => (((j * 29) % 53) - 26) / 26.0);
  close(hd.dot(hd.vjp(host, w, some, r), move), n.vjp_dot_move.value, n.vjp_dot_move, "J'r . d");
  // The dense Jacobian is the same operator.
  const J = hd.jacobian(host, w, some);
  const viaJ = hd.matVec(J, some.n * host.M, host.P, move);
  viaJ.forEach((v, i) => close(v, Jd[i], { atol: 1e-12 }, "J"));
  close(hd.behaviourChange(host, w, hd.addScaled(w, move, 1.0), b.fit), n.behaviour_change.value, n.behaviour_change);
});

each("block iteration finds the shafts an exact SVD finds", (b) => {
  const { host, vec, w, move, fit, options } = b;
  const n = vec.numeric;
  const th = hd.thread(host, w, fit, 16, options);
  n.eigenvalues_16.values.forEach((want, i) => close(th.lam[i], want, n.eigenvalues_16, `lam[${i}]`));
  close(hd.predictedChange(th, move), n.predicted_change.value, n.predicted_change);
  for (let i = 0; i < 16; i++) {
    close(hd.norm(th.V[i]), 1.0, { atol: 1e-10 });
    close(hd.predictedChange(th, hd.shaftMove(th, i, 0.37)), 0.37, { atol: 1e-9 }, "travel is behaviour");
  }
  const tie = hd.tieUp(th, Object.fromEntries(host.treadleNames.map((name) => [name, host.treadlePattern(name, b.fitInputs, hd.forward(host, w, fit))])));
  for (const name of host.treadleNames) {
    close(tie.share[name], n.share_16.values[name], n.share_16, name);
    close(hd.norm(tie.unitMove[name]), n.unit_move_length_16.values[name], n.unit_move_length_16, name);
  }
});

each("with heddles raised the loom is threaded around them", (b) => {
  const { host, vec, w, move, fit, options } = b;
  const n = vec.numeric;
  const pinEnc = host.encode(host.pinInputs(vec.pins));
  const sh = hd.shedOf(host, w, pinEnc);
  assert.equal(sh.openDimensions, vec.exact.open_dimensions);
  close(hd.norm(sh.project(move)), n.projected_move_length.value, n.projected_move_length);
  const th = hd.thread(host, w, fit, 16, { ...options, basis: sh.Q });
  n.eigenvalues_16_in_shed.values.forEach((want, i) => close(th.lam[i], want, n.eigenvalues_16_in_shed, `lam[${i}]`));
  // every shaft already leaves the pinned inputs alone to first order
  for (const v of th.V) assert.ok(Math.max(...hd.jvp(host, w, pinEnc, v).map(Math.abs)) < 1e-8);
  const tie = hd.tieUp(th, Object.fromEntries(host.treadleNames.map((name) => [name, host.treadlePattern(name, b.fitInputs, hd.forward(host, w, fit))])));
  for (const name of host.treadleNames) close(tie.share[name], n.share_16_in_shed.values[name], n.share_16_in_shed, name);
});

each("the proved bound matches the reference and is never exceeded", (b) => {
  const { host, vec, w, move, some } = b;
  const n = vec.numeric;
  const bound = hd.certifiedBound(host, w, some, n.certified_bound.radius);
  n.certified_bound.values.forEach((want, i) => close(bound[i], want, n.certified_bound, "bound"));
  const enc = host.encode(host.heldInputs().slice(0, host.kind === "cloth/1" ? 400 : 200));
  const y0 = hd.forward(host, w, enc);
  const th = hd.thread(host, w, b.fit, 2, { iterations: 30 });
  for (const radius of [0.01, 0.1, 0.5, 2.0]) {
    const limit = hd.certifiedBound(host, w, enc, radius);
    for (const dir of [th.V[0], th.V[1], move]) {
      const y1 = hd.forward(host, hd.addScaled(w, dir, radius / hd.norm(dir)), enc);
      for (let i = 0; i < enc.n; i++) {
        let sq = 0.0;
        for (let c = 0; c < host.M; c++) sq += (y1[i * host.M + c] - y0[i * host.M + c]) ** 2;
        assert.ok(Math.sqrt(sq) <= limit[i], `radius ${radius}: ${Math.sqrt(sq)} above the bound ${limit[i]}`);
      }
    }
  }
  const r = hd.certifiedRadius(host, w, enc, Number(host.tau));
  assert.ok(r > 0 && Math.max(...hd.certifiedBound(host, w, enc, r)) <= Number(host.tau) * Math.sqrt(host.M) * (1 + 1e-9));
});

each("the dispute and the hidden move match the reference", (b) => {
  const { host, vec, w, fit, options } = b;
  const n = vec.numeric;
  const th = hd.thread(host, w, fit, 16, options);
  const found = host.dispute(w, hd.addScaled(w, hd.shaftMove(th, 0, n.dispute.travel), 1.0));
  close(found.worst, n.dispute.worst, n.dispute, "worst");
  assert.equal(found.rows, vec.exact.dispute_rows);
  assert.equal(found.exhaustive, vec.exact.dispute_exhaustive);
  if (found.exhaustive) assert.deepEqual(found.witness, n.dispute.witness);
  const known = host.encode(host.knownInputs()), elsewhere = host.encode(host.elsewhereInputs());
  const wh = hd.addScaled(w, hd.hiddenMove(host, w, known, elsewhere, vec.hide_radius), 1.0);
  close(hd.behaviourChange(host, w, wh, known), n.hidden_move.on_known, n.hidden_move, "on known probes");
  close(hd.behaviourChange(host, w, wh, elsewhere), n.hidden_move.elsewhere, n.hidden_move, "elsewhere");
});

// The same deferrals as weft's own browser test (weft/runtime/aywa/test_invariant_gate.mjs):
// the published gate does not implement these cases, and no loom reaches them.
const BROWSER_DEFERRED = new Set(["reject_pre_cast_radius_boundary", "accept_huge_scale_weight_move", "reject_locked_mask_ulp_tamper", "accept_locked_mask_negzero_normalize", "reject_negative_cost"]);

test("the vendored weft gate passes weft's conformance vectors", () => {
  let ran = 0;
  for (const v of read("../vectors/weft-invariants.json").weight_update_vectors) {
    if (v.locked_mask != null || BROWSER_DEFERRED.has(v.name)) continue;
    const got = hd.verifyWeightUpdateSafety((v.baseline ?? []).map(decode), (v.candidate ?? []).map(decode), decode(v.r_invariant), decode(v.budget), decode(v.cost));
    assert.equal(got.accepted, v.expected_accepted, v.name);
    assert.equal(got.reason, v.expected_reason, v.name);
    const want = decode(v.expected_distance);
    if (Number.isFinite(want)) close(got.distance, want, { atol: 1e-9 }, v.name);
    else assert.equal(got.distance, want, v.name);
    ran += 1;
  }
  assert.ok(ran >= 8);
});

each("a session: light picks weave on release, tense ones need a hold, a refuted claim snaps", (b) => {
  const { host } = b;
  const loom = new hd.Loom({ host });
  const [first, second] = host.treadleNames;
  // Find a travel in each zone by search, so the test does not depend on the host's scale.
  const travelFor = (name, zone) => {
    for (let t = 0.02; t < 3; t *= 1.25) if (loom.assess(loom.moveFor({ [name]: t })).zone === zone) return t;
    throw new Error(`no ${zone} travel for ${name}`);
  };
  const light = travelFor(first, "auto");
  const one = loom.commit(loom.moveFor({ [first]: light }), "release", { [first]: light });
  assert.equal(one.kind, "pick");
  assert.equal(loom.fell.root, one.root);
  assert.equal(loom.fell.budget, host.budget - 1);
  assert.equal(one.claim.exhaustive, host.kind === "text/1");

  const tense = travelFor(second, "shared");
  const move = loom.moveFor({ [second]: tense });
  assert.deepEqual(loom.commit(move, "release", { [second]: tense }), { needs: "shared" });
  assert.equal(loom.events.length, 1, "a gesture that is too light records nothing");
  const two = loom.commit(move, "hold", { [second]: tense });
  assert.equal(two.kind, "pick");
  assert.ok(Math.abs(two.reading.delivered[second].gain - two.reading.delivered[second].promised) < 0.5 * Math.abs(two.reading.delivered[second].promised) + 0.02);

  const held = loom.fell.root;
  const far = loom.moveFor({}, { 0: 4 * Number(host.tau) });
  const snap = loom.commit(far, "knot");
  assert.equal(snap.kind, "snap");
  assert.ok(snap.claim.refuted && snap.claim.worst > Number(host.tau));
  assert.equal(loom.fell.root, held, "a snapped pick does not move the fell");
  assert.equal(loom.fell.budget, host.budget - 3, "and its budget is spent");
  // One input is the whole evidence.
  const cand = hd.addScaled(loom.fell.w, far, 1.0);
  assert.ok(hd.change(host, loom.fell.w, cand, host.encode(host.witnessInputs(snap.claim.witness)))[0] > Number(host.tau));

  loom.unpick(one.root);
  assert.equal(loom.fell.root, one.root);
  assert.deepEqual(loom.path().picks.map((e) => e.seq), [1]);
  loom.events.forEach((e, i) => {
    assert.equal(e.prev, i ? loom.events[i - 1].id : "genesis");
    assert.equal(hd.eventId(e), e.id);
  });
});

each("weft's gate refuses a pick that leaves the ball, and spends nothing", (b) => {
  const host = { ...b.host, rInvariant: 0.02 };
  const loom = new hd.Loom({ host });
  const d = loom.moveFor({}, { 0: 0.5 });
  assert.equal(loom.assess(d).zone, "refused");
  assert.equal(loom.commit(d, "knot").refused.reason, "baseline_distance_exceeded");
  assert.equal(loom.events.length, 0);
  assert.equal(loom.fell.budget, host.budget);
});

each("raising heddles re-threads the loom, and a pick made there leaves them still", (b) => {
  const { host, vec } = b;
  const loom = new hd.Loom({ host });
  const lamBefore = [...loom.threading.lam];
  loom.setPins(vec.pins);
  assert.equal(loom.shed.openDimensions, vec.exact.open_dimensions);
  loom.threading.lam.forEach((l, i) => assert.ok(l <= lamBefore[i] * (1 + 1e-6), "pins never add reach"));
  const [first, second] = host.treadleNames;
  const d = loom.moveFor({ [first]: 0.1, [second]: 0.1 });
  const pinEnc = host.encode(host.pinInputs(loom.pins));
  assert.ok(Math.max(...hd.jvp(host, loom.fell.w, pinEnc, d).map(Math.abs)) < 1e-8);
  const ev = loom.commit(d, "knot", { [first]: 0.1, [second]: 0.1 });
  assert.ok(ev.kind === "pick" || ev.kind === "snap");
  assert.ok(ev.reading.pin_leak < 0.02 * Number(host.tau) + 0.01, `pin leak ${ev.reading.pin_leak}`);
});

test("text: weaving is repeatable, and the same noise shows what a move changed", () => {
  const host = benches.text.host;
  const a = host.weave(host.w0, "the", 80, 3);
  assert.equal(a.text.length, 80);
  assert.deepEqual(host.weave(host.w0, "the", 80, 3), a);
  assert.notEqual(host.weave(host.w0, "the", 80, 4).text, a.text);
  assert.equal(host.contextText(a.contexts[0]), "the");
  const loom = new hd.Loom({ host });
  const moved = host.weave(hd.addScaled(host.w0, loom.moveFor({ vowels: 0.9 }), 1.0), "the", 400, 3).text;
  const vowels = (s) => [...s].filter((ch) => "aeiou".includes(ch)).length;
  assert.ok(vowels(moved) > vowels(host.weave(host.w0, "the", 400, 3).text), "the vowels treadle adds vowels");
});
