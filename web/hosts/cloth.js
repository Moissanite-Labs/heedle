/**
 * cloth/1: a point of the unit square to a colour. 467 weights. A mirror of `ClothHost` in
 * heedle/hosts.py. Inputs are flat Float64Arrays of points: [u0, v0, u1, v1, ...].
 */
import * as hd from "../heedle.js";

const TWO_PI = 2.0 * Math.PI;
const INIT_SCALE = { W1: 1.1, b1: 0.6, W2: 0.9, b2: 0.3, W3: 0.9, b3: 0.2 };
export const FIT_GRID = 10, PIN_LATTICE = 64, DISPUTE_GRID = 96, DISPUTE_REFINE = 4, DISPUTE_STEPS = 24;

/** Cell centres of an n x n lattice; index = row * n + column, u along columns. */
export function grid(n) {
  const pts = new Float64Array(n * n * 2);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      pts[2 * (r * n + c)] = (c + 0.5) / n;
      pts[2 * (r * n + c) + 1] = (r + 0.5) / n;
    }
  }
  return pts;
}

export function makeClothHost(seed = 7) {
  const sh = hd.shapeOf(8, 16, 3, false);
  const draw = hd.mulberry32(seed);
  const w0 = new Float64Array(sh.P);
  for (const name of hd.TENSOR_NAMES) {
    const [lo, hi] = sh.offsets[name];
    for (let i = lo; i < hi; i++) w0[i] = (2.0 * draw() - 1.0) * INIT_SCALE[name];
  }
  const host = {
    ...sh,
    kind: "cloth/1",
    seed,
    w0,
    rInvariant: 2.0,
    tau: "0.5",
    budget: 24,
    cost: 1,
    probes: 64,
    travel: 0.6,
    maxPins: 32,
    pinCount: PIN_LATTICE * PIN_LATTICE,
    treadleNames: ["warm", "light", "tilt", "contrast", "check"],

    encode(pts) {
      const n = pts.length / 2;
      const idx = new Int32Array(n * 8), val = new Float64Array(n * 8);
      for (let i = 0; i < n; i++) {
        const u = pts[2 * i], v = pts[2 * i + 1], o = i * 8;
        for (let f = 0; f < 8; f++) idx[o + f] = f;
        val[o] = 2.0 * u - 1.0;
        val[o + 1] = 2.0 * v - 1.0;
        val[o + 2] = Math.sin(TWO_PI * u);
        val[o + 3] = Math.cos(TWO_PI * u);
        val[o + 4] = Math.sin(TWO_PI * v);
        val[o + 5] = Math.cos(TWO_PI * v);
        val[o + 6] = Math.sin(TWO_PI * (u + v));
        val[o + 7] = Math.sin(TWO_PI * (u - v));
      }
      return { n, nnz: 8, idx, val };
    },
    fitInputs: () => grid(FIT_GRID),
    heldInputs: () => grid(31),
    knownInputs: () => grid(4),
    elsewhereInputs: () => grid(24),
    pinInputs(pins) {
      const list = hd.normalisePins(pins, host.pinCount);
      const pts = new Float64Array(list.length * 2);
      list.forEach((p, i) => {
        pts[2 * i] = ((p % PIN_LATTICE) + 0.5) / PIN_LATTICE;
        pts[2 * i + 1] = (Math.floor(p / PIN_LATTICE) + 0.5) / PIN_LATTICE;
      });
      return pts;
    },
    freshInputs(seedBytes, n) {
      const pts = new Float64Array(n * 2);
      for (let i = 0; i < n; i++) {
        pts[2 * i] = hd.seededUint32(seedBytes, i, 0) / 4294967296;
        pts[2 * i + 1] = hd.seededUint32(seedBytes, i, 1) / 4294967296;
      }
      return pts;
    },
    treadlePattern(name, pts, y) {
      const n = pts.length / 2, M = 3;
      const c = new Float64Array(n * M);
      let mean = 0.0;
      if (name === "contrast") {
        for (let i = 0; i < n; i++) mean += (y[i * M] + y[i * M + 1] + y[i * M + 2]) / M;
        mean /= n;
      }
      for (let i = 0; i < n; i++) {
        const u = pts[2 * i], v = pts[2 * i + 1];
        let per;
        if (name === "warm") per = [1.0, 0.0, -1.0];
        else if (name === "light") per = [1.0, 1.0, 1.0];
        else if (name === "tilt") per = [2.0 * u - 1.0, 2.0 * u - 1.0, 2.0 * u - 1.0];
        else if (name === "contrast") {
          const light = (y[i * M] + y[i * M + 1] + y[i * M + 2]) / M - mean;
          per = [light, light, light];
        } else if (name === "check") {
          const s = Math.sin(2.0 * TWO_PI * u) * Math.sin(2.0 * TWO_PI * v);
          per = [s, s, s];
        } else throw new Error(`unknown treadle ${name}`);
        for (let k = 0; k < M; k++) c[i * M + k] = per[k];
      }
      return hd.unitRms(c);
    },
    /** A bounded search: read a whole lattice, then climb from the best cells. A sampled reading. */
    dispute(w0_, w1) {
      const pts = grid(DISPUTE_GRID);
      const ch = hd.change(host, w0_, w1, host.encode(pts));
      let rows = ch.length;
      const order = Array.from(ch.keys()).sort((a, b) => ch[b] - ch[a] || a - b).slice(0, DISPUTE_REFINE);
      let bestVal = ch[order[0]], bestPt = [pts[2 * order[0]], pts[2 * order[0] + 1]];
      const clip = (v) => Math.min(1.0, Math.max(0.0, v));
      for (const at of order) {
        let p = [pts[2 * at], pts[2 * at + 1]], val = ch[at], step = 0.5 / DISPUTE_GRID;
        for (let it = 0; it < DISPUTE_STEPS; it++) {
          const cand = new Float64Array([clip(p[0] + step), clip(p[1]), clip(p[0] - step), clip(p[1]), clip(p[0]), clip(p[1] + step), clip(p[0]), clip(p[1] - step)]);
          const vals = hd.change(host, w0_, w1, host.encode(cand));
          rows += 4;
          let j = 0;
          for (let k = 1; k < 4; k++) if (vals[k] > vals[j]) j = k;
          if (vals[j] > val) {
            p = [cand[2 * j], cand[2 * j + 1]];
            val = vals[j];
          } else step *= 0.5;
        }
        if (val > bestVal) {
          bestVal = val;
          bestPt = p;
        }
      }
      return { worst: bestVal, witness: bestPt, rows, exhaustive: false };
    },
    witnessInputs: (witness) => Float64Array.from(witness),
    descriptor: () => ({ kind: "cloth/1", seed, w0_root: hd.weightRoot(host, w0) }),
  };
  return host;
}
