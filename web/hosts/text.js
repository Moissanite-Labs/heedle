/**
 * text/1: three characters to the next one. 4,700 weights, trained on this repository's own
 * documents. A mirror of `TextHost` in heedle/hosts.py. Inputs are Int32Arrays of contexts;
 * a context is c0 * V^2 + c1 * V + c2, with c2 the most recent character.
 *
 * `data` is the parsed hosts/text-v1.json.
 */
import * as hd from "../heedle.js";

export const ALPHABET = "abcdefghijklmnopqrstuvwxyz .";
const V = 28, K = 3, VOWELS = "aeiou";

export function makeTextHost(data) {
  if (data.schema !== "heedle-host/text/1" || data.alphabet !== ALPHABET) throw new Error("not a text/1 host file");
  const sh = hd.shapeOf(K * V, data.hidden, V, true);
  const w0 = new Float64Array(sh.P);
  for (const name of hd.TENSOR_NAMES) w0.set(data.weights[name].flat(), sh.offsets[name][0]);
  const held = data.contexts.held, cumulative = [];
  data.contexts.held_counts.reduce((sum, c) => (cumulative.push(sum + c), sum + c), 0);
  const total = cumulative[cumulative.length - 1];
  const everything = Int32Array.from({ length: V ** K }, (_, i) => i);
  let everyEnc = null;
  let seen = { w: null, y: null };

  const host = {
    ...sh,
    kind: "text/1",
    name: data.name,
    w0,
    radius: data.r_invariant,
    tau: data.tau,
    budget: 24,
    cost: 1,
    probes: 64,
    travel: 1.0,
    maxPins: 6,
    pinCount: V ** K,
    treadleNames: ["vowels", "spaces", "stops", "echo", "sharp"],
    V,
    K,

    contextOf(s) {
      if (s.length !== K) throw new Error(`a context is ${K} characters`);
      let c = 0;
      for (const ch of s) {
        const at = ALPHABET.indexOf(ch);
        if (at < 0) throw new Error("a context uses the 28 symbols of the alphabet");
        c = c * V + at;
      }
      return c;
    },
    contextText(c) {
      return ALPHABET[Math.floor(c / (V * V)) % V] + ALPHABET[Math.floor(c / V) % V] + ALPHABET[c % V];
    },
    encode(contexts) {
      const n = contexts.length;
      const idx = new Int32Array(n * K), val = new Float64Array(n * K).fill(1.0);
      for (let i = 0; i < n; i++) {
        const c = contexts[i];
        idx[i * K] = Math.floor(c / (V * V)) % V;
        idx[i * K + 1] = V + (Math.floor(c / V) % V);
        idx[i * K + 2] = 2 * V + (c % V);
      }
      return { n, nnz: K, idx, val };
    },
    fitInputs: () => Int32Array.from(data.contexts.fit),
    heldInputs: () => Int32Array.from(held),
    knownInputs: () => Int32Array.from(data.contexts.known.slice(0, 8)),
    elsewhereInputs: () => Int32Array.from(held.slice(0, 200)),
    pinInputs: (pins) => Int32Array.from(hd.normalisePins(pins, host.pinCount)),
    /** Held-out contexts, drawn in proportion to how often the corpus uses them. Integers only. */
    freshInputs(seedBytes, n) {
      const out = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        const r = hd.seededUint32(seedBytes, i) % total;
        let lo = 0, hi = cumulative.length - 1;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if (cumulative[mid] > r) hi = mid;
          else lo = mid + 1;
        }
        out[i] = held[lo];
      }
      return out;
    },
    treadlePattern(name, contexts, y) {
      const n = contexts.length;
      const c = new Float64Array(n * V);
      for (let i = 0; i < n; i++) {
        const o = i * V;
        if (name === "vowels") for (const ch of VOWELS) c[o + ALPHABET.indexOf(ch)] = 1.0;
        else if (name === "spaces") c[o + ALPHABET.indexOf(" ")] = 1.0;
        else if (name === "stops") c[o + ALPHABET.indexOf(".")] = 1.0;
        else if (name === "echo") c[o + (contexts[i] % V)] = 1.0;
        else if (name === "sharp") for (let k = 0; k < V; k++) c[o + k] = y[o + k];
        else throw new Error(`unknown treadle ${name}`);
        let mean = 0.0;
        for (let k = 0; k < V; k++) mean += c[o + k];
        mean /= V;
        for (let k = 0; k < V; k++) c[o + k] -= mean;
      }
      return hd.unitRms(c);
    },
    /** Every context there is, read once. Exhaustive over the host's domain, in float64. */
    dispute(w0_, w1) {
      everyEnc ??= host.encode(everything);
      // The outputs of the last candidate are kept: when it became the fell, half the work is done.
      // The cache is keyed on the array itself, so it relies on weights never being changed in
      // place. A loom never does that: every candidate is a new array.
      const y0 = seen.w === w0_ ? seen.y : hd.forward(host, w0_, everyEnc);
      const y1 = hd.forward(host, w1, everyEnc);
      seen = { w: w1, y: y1 };
      let worst = 0, top = -1.0;
      for (let i = 0; i < everything.length; i++) {
        let sq = 0.0;
        for (let k = 0; k < V; k++) sq += (y1[i * V + k] - y0[i * V + k]) ** 2;
        if (sq > top) {
          top = sq;
          worst = i;
        }
      }
      return { worst: Math.sqrt(top / V), witness: [worst], rows: everything.length, exhaustive: true };
    },
    provedInputs: () => everything,
    witnessInputs(witness) {
      if (witness.length !== 1 || !Number.isInteger(witness[0]) || witness[0] < 0 || witness[0] >= host.pinCount) throw new Error("witness outside the host's domain");
      return Int32Array.from(witness);
    },
    descriptor: () => ({ kind: "text/1", name: data.name, w0_root: hd.weightRoot(host, w0) }),

    /**
     * Sample `length` characters after `tail`, one uniform draw per character from a seeded
     * generator, so two sets of weights are compared on the same noise. For the eye only.
     * Returns the text and, per character, the context it was drawn in.
     */
    weave(w, tail, length, seed) {
      const draw = hd.mulberry32(seed);
      let text = tail.slice(-K).padStart(K, " ");
      const one = new Int32Array(1), y = new Float64Array(V);
      let out = "";
      const contexts = new Int32Array(length);
      for (let i = 0; i < length; i++) {
        one[0] = host.contextOf(text.slice(-K));
        contexts[i] = one[0];
        hd.forward(host, w, host.encode(one), y);
        let top = -Infinity;
        for (let k = 0; k < V; k++) top = Math.max(top, y[k]);
        let sum = 0.0;
        for (let k = 0; k < V; k++) sum += Math.exp(y[k] - top);
        let acc = 0.0, pick = V - 1;
        const u = draw() * sum;
        for (let k = 0; k < V; k++) {
          acc += Math.exp(y[k] - top);
          if (acc > u) {
            pick = k;
            break;
          }
        }
        out += ALPHABET[pick];
        text += ALPHABET[pick];
      }
      return { text: out, contexts };
    },
  };
  return host;
}
