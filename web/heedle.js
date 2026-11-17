/**
 * Heedle core for the browser and for Node: the mathematics of steering a small model by
 * hand, for any host. A mirror of `heedle/core.py`, which is the reference. `vectors/` pins
 * the two together (written by tools/make_vectors.py, read by web/test_heedle.mjs).
 *
 * A host is a three-layer tanh network with sparse feature vectors. Weights are one
 * Float64Array of length P in the order W1, b1, W2, b2, W3, b3 (row-major). An encoded batch
 * of inputs is { n, nnz, idx: Int32Array(n * nnz), val: Float64Array(n * nnz) }.
 *
 * Nothing here forms the Jacobian of the fit probes. Shafts are found with J v and J' r only,
 * so a host of several thousand weights can be threaded in a browser.
 */
import { verifyWeightUpdateSafety } from "./vendor/invariant-gate.js";

export { verifyWeightUpdateSafety };
export const TENSOR_NAMES = ["W1", "b1", "W2", "b2", "W3", "b3"];
export const GAMMA = 1.0;
const CBC_DOMAIN = "WEFT-CBC-V1:";
const DOMAIN = "HEEDLE-V1:";

// ---------------------------------------------------------------------------------------
// Deterministic numbers
// ---------------------------------------------------------------------------------------

/** mulberry32: integer arithmetic only, so every draw matches the Python reference bit for bit. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function draw() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------------------
// Shape and the host network
// ---------------------------------------------------------------------------------------

/** Sizes of a host: F features, H hidden units in each layer, M outputs. */
export function shapeOf(F, H, M, centre = false) {
  const shapes = { W1: [H, F], b1: [H], W2: [H, H], b2: [H], W3: [M, H], b3: [M] };
  const offsets = {};
  let at = 0;
  for (const name of TENSOR_NAMES) {
    const size = shapes[name].reduce((a, b) => a * b, 1);
    offsets[name] = [at, at + size];
    at += size;
  }
  return { F, H, M, centre, shapes, offsets, P: at };
}

/** Hidden activations of one input, written into h1 and h2. */
function hidden(sh, w, enc, i, h1, h2) {
  const { F, H, offsets: o } = sh;
  const oW1 = o.W1[0], ob1 = o.b1[0], oW2 = o.W2[0], ob2 = o.b2[0];
  const nnz = enc.nnz, eo = i * nnz;
  for (let j = 0; j < H; j++) {
    let a = w[ob1 + j];
    const wo = oW1 + j * F;
    for (let t = 0; t < nnz; t++) a += w[wo + enc.idx[eo + t]] * enc.val[eo + t];
    h1[j] = Math.tanh(a);
  }
  for (let j = 0; j < H; j++) {
    let a = w[ob2 + j];
    const wo = oW2 + j * H;
    for (let k = 0; k < H; k++) a += w[wo + k] * h1[k];
    h2[j] = Math.tanh(a);
  }
}

function centreRow(y, o, M) {
  let mean = 0.0;
  for (let c = 0; c < M; c++) mean += y[o + c];
  mean /= M;
  for (let c = 0; c < M; c++) y[o + c] -= mean;
}

/** Outputs, flat (n * M). The hot path: pass `out` and nothing is allocated per call. */
export function forward(sh, w, enc, out = null) {
  const { H, M, offsets: o } = sh;
  const y = out ?? new Float64Array(enc.n * M);
  const h1 = new Float64Array(H), h2 = new Float64Array(H);
  const oW3 = o.W3[0], ob3 = o.b3[0];
  for (let i = 0; i < enc.n; i++) {
    hidden(sh, w, enc, i, h1, h2);
    for (let c = 0; c < M; c++) {
      let a = w[ob3 + c];
      const wo = oW3 + c * H;
      for (let j = 0; j < H; j++) a += w[wo + j] * h2[j];
      y[i * M + c] = a;
    }
    if (sh.centre) centreRow(y, i * M, M);
  }
  return y;
}

/** J d: the first-order output change of a weight move d, flat (n * M). */
export function jvp(sh, w, enc, d, out = null) {
  const { F, H, M, offsets: o } = sh;
  const y = out ?? new Float64Array(enc.n * M);
  const h1 = new Float64Array(H), h2 = new Float64Array(H), t1 = new Float64Array(H), t2 = new Float64Array(H);
  const oW1 = o.W1[0], ob1 = o.b1[0], oW2 = o.W2[0], ob2 = o.b2[0], oW3 = o.W3[0], ob3 = o.b3[0];
  const nnz = enc.nnz;
  for (let i = 0; i < enc.n; i++) {
    hidden(sh, w, enc, i, h1, h2);
    const eo = i * nnz;
    for (let j = 0; j < H; j++) {
      let a = d[ob1 + j];
      const wo = oW1 + j * F;
      for (let t = 0; t < nnz; t++) a += d[wo + enc.idx[eo + t]] * enc.val[eo + t];
      t1[j] = (1.0 - h1[j] * h1[j]) * a;
    }
    for (let j = 0; j < H; j++) {
      let a = d[ob2 + j];
      const wo = oW2 + j * H;
      for (let k = 0; k < H; k++) a += w[wo + k] * t1[k] + d[wo + k] * h1[k];
      t2[j] = (1.0 - h2[j] * h2[j]) * a;
    }
    for (let c = 0; c < M; c++) {
      let a = d[ob3 + c];
      const wo = oW3 + c * H;
      for (let j = 0; j < H; j++) a += w[wo + j] * t2[j] + d[wo + j] * h2[j];
      y[i * M + c] = a;
    }
    if (sh.centre) centreRow(y, i * M, M);
  }
  return y;
}

/** J' r: the weight direction that an output pattern r, flat (n * M), pulls on. */
export function vjp(sh, w, enc, r, out = null) {
  const { F, H, M, P, offsets: o } = sh;
  const g = out ?? new Float64Array(P);
  g.fill(0.0);
  const h1 = new Float64Array(H), h2 = new Float64Array(H), d1 = new Float64Array(H), d2 = new Float64Array(H);
  const row = new Float64Array(M);
  const oW1 = o.W1[0], ob1 = o.b1[0], oW2 = o.W2[0], ob2 = o.b2[0], oW3 = o.W3[0], ob3 = o.b3[0];
  const nnz = enc.nnz;
  for (let i = 0; i < enc.n; i++) {
    hidden(sh, w, enc, i, h1, h2);
    for (let c = 0; c < M; c++) row[c] = r[i * M + c];
    if (sh.centre) centreRow(row, 0, M);
    d2.fill(0.0);
    for (let c = 0; c < M; c++) {
      const rc = row[c];
      if (rc === 0.0) continue;
      const wo = oW3 + c * H;
      for (let j = 0; j < H; j++) {
        g[wo + j] += rc * h2[j];
        d2[j] += rc * w[wo + j];
      }
      g[ob3 + c] += rc;
    }
    d1.fill(0.0);
    for (let j = 0; j < H; j++) {
      const v = d2[j] * (1.0 - h2[j] * h2[j]);
      const wo = oW2 + j * H;
      for (let k = 0; k < H; k++) {
        g[wo + k] += v * h1[k];
        d1[k] += v * w[wo + k];
      }
      g[ob2 + j] += v;
    }
    const eo = i * nnz;
    for (let k = 0; k < H; k++) {
      const v = d1[k] * (1.0 - h1[k] * h1[k]);
      const wo = oW1 + k * F;
      for (let t = 0; t < nnz; t++) g[wo + enc.idx[eo + t]] += v * enc.val[eo + t];
      g[ob1 + k] += v;
    }
  }
  return g;
}

/** The dense Jacobian, (n * M) x P row-major. For small input sets: pins, known probes, tests. */
export function jacobian(sh, w, enc) {
  const rows = enc.n * sh.M;
  const J = new Float64Array(rows * sh.P);
  const one = { n: 1, nnz: enc.nnz, idx: null, val: null };
  const r = new Float64Array(sh.M);
  for (let i = 0; i < enc.n; i++) {
    one.idx = enc.idx.subarray(i * enc.nnz, (i + 1) * enc.nnz);
    one.val = enc.val.subarray(i * enc.nnz, (i + 1) * enc.nnz);
    for (let c = 0; c < sh.M; c++) {
      r.fill(0.0);
      r[c] = 1.0;
      vjp(sh, w, one, r, J.subarray((i * sh.M + c) * sh.P, (i * sh.M + c + 1) * sh.P));
    }
  }
  return J;
}

/** Per-input change: root mean square over the outputs of f1 - f0. */
export function change(sh, w0, w1, enc) {
  const y0 = forward(sh, w0, enc), y1 = forward(sh, w1, enc);
  const out = new Float64Array(enc.n);
  for (let i = 0; i < enc.n; i++) {
    let s = 0.0;
    for (let c = 0; c < sh.M; c++) {
      const d = y1[i * sh.M + c] - y0[i * sh.M + c];
      s += d * d;
    }
    out[i] = Math.sqrt(s / sh.M);
  }
  return out;
}

/** D_X of the calibration theory: root mean square of all output differences on the inputs. */
export function behaviourChange(sh, w0, w1, enc) {
  const y0 = forward(sh, w0, enc), y1 = forward(sh, w1, enc);
  let s = 0.0;
  for (let i = 0; i < y0.length; i++) s += (y1[i] - y0[i]) * (y1[i] - y0[i]);
  return Math.sqrt(s / y0.length);
}

/** Largest per-input root mean square of a flat (n * M) output change. */
export function worstRow(y, M) {
  let worst = 0.0;
  for (let i = 0; i < y.length / M; i++) {
    let s = 0.0;
    for (let c = 0; c < M; c++) s += y[i * M + c] * y[i * M + c];
    worst = Math.max(worst, Math.sqrt(s / M));
  }
  return worst;
}

// ---------------------------------------------------------------------------------------
// Small linear algebra
// ---------------------------------------------------------------------------------------

export function dot(a, b) {
  let s = 0.0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a) {
  return Math.sqrt(dot(a, a));
}

export function addScaled(a, b, s) {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + s * b[i];
  return out;
}

/** y = A v for a row-major (rows x cols) matrix. */
export function matVec(A, rows, cols, v, out) {
  const y = out ?? new Float64Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = 0.0;
    const o = r * cols;
    for (let c = 0; c < cols; c++) s += A[o + c] * v[c];
    y[r] = s;
  }
  return y;
}

/** y = A' z for a row-major (rows x cols) matrix. */
export function matTVec(A, rows, cols, z, out) {
  const y = out ?? new Float64Array(cols);
  y.fill(0.0);
  for (let r = 0; r < rows; r++) {
    const s = z[r];
    if (s === 0.0) continue;
    const o = r * cols;
    for (let c = 0; c < cols; c++) y[c] += s * A[o + c];
  }
  return y;
}

/**
 * Eigen-decomposition of a small symmetric matrix by cyclic Jacobi rotations.
 * `S` is row-major (n x n) and is not changed. Returns eigenvalues in descending order and
 * the matching eigenvectors as the rows of `vectors`.
 */
export function symmetricEigen(S, n) {
  const A = Float64Array.from(S);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1.0;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0.0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
    let diag = 0.0;
    for (let p = 0; p < n; p++) diag += A[p * n + p] * A[p * n + p];
    if (off <= 1e-30 * (diag + 1e-300)) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (A[q * n + q] - A[p * n + p]) / (2.0 * apq);
        const t = Math.sign(theta || 1.0) / (Math.abs(theta) + Math.sqrt(theta * theta + 1.0));
        const c = 1.0 / Math.sqrt(t * t + 1.0);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => A[b * n + b] - A[a * n + a]);
  const values = new Float64Array(n);
  const vectors = new Float64Array(n * n);
  order.forEach((src, dst) => {
    values[dst] = A[src * n + src];
    for (let k = 0; k < n; k++) vectors[dst * n + k] = V[k * n + src];
  });
  return { values, vectors };
}

/** Largest singular value of a row-major (rows x cols) matrix, from the eigenvalues of A A'. */
export function largestSingularValue(A, rows, cols) {
  const S = new Float64Array(rows * rows);
  for (let i = 0; i < rows; i++) {
    for (let j = i; j < rows; j++) {
      let s = 0.0;
      for (let k = 0; k < cols; k++) s += A[i * cols + k] * A[j * cols + k];
      S[i * rows + j] = s;
      S[j * rows + i] = s;
    }
  }
  return Math.sqrt(Math.max(0.0, symmetricEigen(S, rows).values[0]));
}

/**
 * Orthonormal rows spanning the rows of A (rows x cols), by modified Gram-Schmidt run twice.
 * A row is dropped when what is left of it is below rtol times its own length. Returns an
 * array of Float64Array rows. The same algorithm as `row_space_basis` in the reference.
 */
export function rowSpaceBasis(A, rows, cols, rtol = 1e-10) {
  const basis = [];
  for (let r = 0; r < rows; r++) {
    const v = A.slice(r * cols, (r + 1) * cols);
    const scale = norm(v);
    if (scale === 0.0) continue;
    for (let pass = 0; pass < 2; pass++) {
      for (const q of basis) {
        const s = dot(q, v);
        for (let c = 0; c < cols; c++) v[c] -= s * q[c];
      }
    }
    const nv = norm(v);
    if (nv > rtol * scale) {
      for (let c = 0; c < cols; c++) v[c] /= nv;
      basis.push(v);
    }
  }
  return basis;
}

function orthonormaliseRows(vectors) {
  const kept = [];
  for (const v of vectors) {
    for (let pass = 0; pass < 2; pass++) {
      for (const q of kept) {
        const s = dot(q, v);
        for (let c = 0; c < v.length; c++) v[c] -= s * q[c];
      }
    }
    const nv = norm(v);
    if (nv > 1e-14) {
      for (let c = 0; c < v.length; c++) v[c] /= nv;
      kept.push(v);
    }
  }
  return kept;
}

// ---------------------------------------------------------------------------------------
// Shafts, treadles and the tie-up
// ---------------------------------------------------------------------------------------

/**
 * The linearisation of a host at `w` on the encoded fit probes: the first `shafts` eigenpairs
 * of G = J'J/(n m), by block power iteration on J J' with a Rayleigh-Ritz step. Only J v and
 * J' r are used. The reference takes an exact SVD; the test checks that the eigenvalues agree.
 *
 *   lam[i]   eigenvalue of shaft i
 *   V[i]     the shaft: a unit direction in weight space (largest entry positive)
 *   U[i]     the unit-length output direction it makes; U[i] * sqrt(rows) is its pattern
 *
 * `warm` may be a previous threading, whose block starts the iteration. `basis` may hold the
 * orthonormal rows of a shed: the shafts are then those of G restricted to the shed, so every
 * shaft already leaves the pinned inputs alone to first order.
 */
export function thread(sh, w, enc, shafts, { iterations = 200, oversample = 8, warm = null, basis = null } = {}) {
  const rows = enc.n * sh.M;
  const block = Math.min(shafts + oversample, rows);
  // J restricted to the shed is J - (J Q') Q, so J_s J_s' = J J' - B B' with B = J Q'.
  const B = basis && basis.length ? basis.map((q) => jvp(sh, w, enc, q)) : [];
  const v = new Float64Array(sh.P);
  const op = (z) => {
    const y = jvp(sh, w, enc, vjp(sh, w, enc, z, v));
    for (const b of B) {
      const s = dot(b, z);
      for (let t = 0; t < rows; t++) y[t] -= s * b[t];
    }
    return y;
  };
  let Z = [];
  if (warm && warm.block && warm.block.length === block && warm.rows === rows) {
    Z = warm.block.map((z) => Float64Array.from(z));
  } else {
    const draw = mulberry32(0x5eed);
    for (let k = 0; k < block; k++) {
      const z = new Float64Array(rows);
      for (let i = 0; i < rows; i++) z[i] = 2.0 * draw() - 1.0;
      Z.push(z);
    }
  }
  Z = orthonormaliseRows(Z.map(op));
  for (let it = 0; it < iterations; it++) Z = orthonormaliseRows(Z.map(op));
  const k = Z.length;
  const GZ = Z.map(op);
  const S = new Float64Array(k * k);
  for (let i = 0; i < k; i++) {
    for (let j = i; j < k; j++) {
      const s = 0.5 * (dot(Z[i], GZ[j]) + dot(Z[j], GZ[i]));
      S[i * k + j] = s;
      S[j * k + i] = s;
    }
  }
  const eig = symmetricEigen(S, k);
  const rotated = [];
  for (let i = 0; i < k; i++) {
    const u = new Float64Array(rows);
    for (let j = 0; j < k; j++) {
      const c = eig.vectors[i * k + j];
      const src = Z[j];
      for (let t = 0; t < rows; t++) u[t] += c * src[t];
    }
    rotated.push(u);
  }
  const lam = [], V = [], U = [];
  for (let i = 0; i < Math.min(shafts, k); i++) {
    const u = rotated[i];
    const x = vjp(sh, w, enc, u);
    if (basis) for (const q of basis) {
      const s = dot(q, x);
      for (let t = 0; t < sh.P; t++) x[t] -= s * q[t];
    }
    const s = norm(x); // the singular value: J_s' u = s v
    let big = 0;
    for (let t = 1; t < sh.P; t++) if (Math.abs(x[t]) > Math.abs(x[big])) big = t;
    const sign = x[big] < 0 ? -1.0 : 1.0;
    for (let t = 0; t < sh.P; t++) x[t] = (sign * x[t]) / (s || 1.0);
    // u is recomputed from v, so J v = s u holds to rounding whatever the iteration reached.
    const ju = jvp(sh, w, enc, x);
    const js = norm(ju);
    for (let t = 0; t < rows; t++) ju[t] /= js || 1.0;
    lam.push((js * js) / rows);
    V.push(x);
    U.push(ju);
  }
  return { sh, w: Float64Array.from(w), enc, rows, lam, V, U, block: rotated };
}

/** First-order output change of a move on a threading's fit probes, flat. */
export function firstOrder(threading, d) {
  return jvp(threading.sh, threading.w, threading.enc, d);
}

/** First-order behaviour change sqrt(d'Gd) of a move on a threading's fit probes. */
export function predictedChange(threading, d) {
  const y = firstOrder(threading, d);
  return Math.sqrt(dot(y, y) / threading.rows);
}

/** The move along shaft i whose first-order behaviour change is |travel|. */
export function shaftMove(threading, i, travel) {
  const s = travel / Math.sqrt(threading.lam[i]);
  return threading.V[i].map((x) => s * x);
}

export function unitRms(c) {
  const r = Math.sqrt(dot(c, c) / c.length);
  return r < 1e-300 ? new Float64Array(c.length) : c.map((v) => v / r);
}

/**
 * Treadles tied to the shafts of a threading. For a pattern c, alpha[i] = u_i'c / sqrt(rows)
 * and the move for travel t is t * sum_i alpha[i] v_i / sqrt(lam[i]). `share` is the part
 * of the pattern the shafts can carry: the first-order gain along c per unit of travel.
 */
export function tieUp(threading, patterns) {
  const root = Math.sqrt(threading.rows);
  const K = threading.V.length, P = threading.sh.P;
  const alpha = {}, share = {}, unitMove = {};
  for (const [name, c] of Object.entries(patterns)) {
    const a = new Float64Array(K), d = new Float64Array(P);
    let total = 0.0;
    for (let i = 0; i < K; i++) {
      a[i] = dot(threading.U[i], c) / root;
      total += a[i] * a[i];
      const s = a[i] / Math.sqrt(threading.lam[i]);
      for (let t = 0; t < P; t++) d[t] += s * threading.V[i][t];
    }
    alpha[name] = a;
    share[name] = total;
    unitMove[name] = d;
  }
  return { K, names: Object.keys(patterns), patterns, alpha, share, unitMove };
}

/** How much of `pattern` an output change contains: <dy, c> / <c, c>. */
export function gainAlong(dy, pattern) {
  return dot(dy, pattern) / dot(pattern, pattern);
}

// ---------------------------------------------------------------------------------------
// Pins and the shed
// ---------------------------------------------------------------------------------------

/** The moves that leave the pinned inputs alone to first order. `pinEnc` may be null. */
export function shedOf(sh, w, pinEnc) {
  const Q = pinEnc && pinEnc.n ? rowSpaceBasis(jacobian(sh, w, pinEnc), pinEnc.n * sh.M, sh.P) : [];
  return {
    Q,
    openDimensions: sh.P - Q.length,
    project(d) {
      const out = Float64Array.from(d);
      for (const q of Q) {
        const s = dot(q, d);
        for (let t = 0; t < sh.P; t++) out[t] -= s * q[t];
      }
      return out;
    },
  };
}

/**
 * A move of length `radius` that the known probes cannot see to first order and that changes
 * as much as it can elsewhere: power iteration on the elsewhere Jacobian inside the null
 * space of the known probes' Jacobian.
 */
export function hiddenMove(sh, w, known, elsewhere, radius, iterations = 60) {
  const Q = rowSpaceBasis(jacobian(sh, w, known), known.n * sh.M, sh.P);
  const project = (v) => {
    for (const q of Q) {
      const s = dot(q, v);
      for (let t = 0; t < sh.P; t++) v[t] -= s * q[t];
    }
    const nv = norm(v);
    for (let t = 0; t < sh.P; t++) v[t] /= nv;
    return v;
  };
  const draw = mulberry32(20261117);
  let v = new Float64Array(sh.P);
  for (let i = 0; i < sh.P; i++) v[i] = 2.0 * draw() - 1.0;
  v = project(v);
  for (let it = 0; it < iterations; it++) v = project(vjp(sh, w, elsewhere, jvp(sh, w, elsewhere, v)));
  let big = 0;
  for (let t = 1; t < sh.P; t++) if (Math.abs(v[t]) > Math.abs(v[big])) big = t;
  const sign = v[big] < 0 ? -radius : radius;
  return v.map((x) => sign * x);
}

// ---------------------------------------------------------------------------------------
// The proved bound
// ---------------------------------------------------------------------------------------

/**
 * Theorem 1 of the calibration theory with slope 1: for every move of length at most
 * `radius` and every input, the output change (Euclidean length) is at most the returned
 * bound. One value per input.
 */
export function certifiedBound(sh, w0, enc, radius) {
  const { H, M, offsets: o } = sh;
  const s2 = largestSingularValue(w0.subarray(o.W2[0], o.W2[1]), H, H);
  const s3 = largestSingularValue(w0.subarray(o.W3[0], o.W3[1]), M, H);
  const out = new Float64Array(enc.n);
  const h1 = new Float64Array(H), h2 = new Float64Array(H);
  const g = GAMMA, r = radius;
  for (let i = 0; i < enc.n; i++) {
    hidden(sh, w0, enc, i, h1, h2);
    let xx = 0.0;
    for (let t = 0; t < enc.nnz; t++) xx += enc.val[i * enc.nnz + t] ** 2;
    const a0 = Math.sqrt(xx + 1.0), a1 = Math.sqrt(dot(h1, h1) + 1.0), a2 = Math.sqrt(dot(h2, h2) + 1.0);
    const c = Math.sqrt((g * g * s3 * s2 * a0) ** 2 + (g * s3 * a1) ** 2 + a2 ** 2);
    const q = (g * g * (s2 + s3) * a0 + g * a1) / 2.0;
    const k = (g * g * a0) / (3.0 * Math.sqrt(3.0));
    out[i] = r * c + r * r * q + r * r * r * k;
  }
  return out;
}

/** The largest radius at which the proved bound keeps every input of `enc` within `tau`. */
export function certifiedRadius(sh, w0, enc, tau) {
  const limit = tau * Math.sqrt(sh.M);
  const worst = (r) => certifiedBound(sh, w0, enc, r).reduce((a, b) => Math.max(a, b), 0.0);
  let lo = 0.0, hi = 1.0;
  for (let i = 0; i < 60; i++) {
    const mid = 0.5 * (lo + hi);
    if (worst(mid) <= limit) lo = mid;
    else hi = mid;
  }
  return lo;
}

// ---------------------------------------------------------------------------------------
// SHA-256 and commitments
// ---------------------------------------------------------------------------------------

const K256 = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

/** SHA-256 of a Uint8Array, as a Uint8Array. Synchronous, so a pick can be rooted inside a frame. */
export function sha256(bytes) {
  const h = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  const len = bytes.length;
  const total = ((len + 9 + 63) >> 6) << 6;
  const m = new Uint8Array(total);
  m.set(bytes);
  m[len] = 0x80;
  const view = new DataView(m.buffer);
  view.setUint32(total - 8, Math.floor(len / 0x20000000), false);
  view.setUint32(total - 4, (len << 3) >>> 0, false);
  const w = new Uint32Array(64);
  for (let o = 0; o < total; o += 64) {
    for (let i = 0; i < 16; i++) w[i] = view.getUint32(o + 4 * i, false);
    for (let i = 16; i < 64; i++) {
      const a = w[i - 15], b = w[i - 2];
      const s0 = ((a >>> 7) | (a << 25)) ^ ((a >>> 18) | (a << 14)) ^ (a >>> 3);
      const s1 = ((b >>> 17) | (b << 15)) ^ ((b >>> 19) | (b << 13)) ^ (b >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, hh] = h;
    for (let i = 0; i < 64; i++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const t1 = (hh + S1 + ch + K256[i] + w[i]) >>> 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (S0 + maj) >>> 0;
      hh = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    h[0] = (h[0] + a) >>> 0; h[1] = (h[1] + b) >>> 0; h[2] = (h[2] + c) >>> 0; h[3] = (h[3] + d) >>> 0;
    h[4] = (h[4] + e) >>> 0; h[5] = (h[5] + f) >>> 0; h[6] = (h[6] + g) >>> 0; h[7] = (h[7] + hh) >>> 0;
  }
  const out = new Uint8Array(32);
  const ov = new DataView(out.buffer);
  for (let i = 0; i < 8; i++) ov.setUint32(4 * i, h[i], false);
  return out;
}

const encoder = new TextEncoder();
const ascii = (s) => encoder.encode(s);

export function toHex(bytes) {
  let s = "";
  for (const b of bytes) s += b.toString(16).padStart(2, "0");
  return s;
}

function concatBytes(...parts) {
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(total);
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

/** JSON with sorted keys and no spaces: the bytes Python's json.dumps(sort_keys=True, separators=(",", ":")) gives for strings, integers and lists. */
export function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((k) => `${JSON.stringify(k)}:${canonicalJson(value[k])}`).join(",")}}`;
  }
  if (typeof value === "number" && !Number.isInteger(value)) throw new Error("only integers and strings are hashed");
  return JSON.stringify(value);
}

/** Little-endian float64 bytes of a weight vector slice. */
function f64Bytes(values) {
  const out = new Uint8Array(values.length * 8);
  const view = new DataView(out.buffer);
  for (let i = 0; i < values.length; i++) view.setFloat64(8 * i, values[i], true);
  return out;
}

/** The weight root of committed-behaviour-change v1 over the host's six tensors, float64. */
export function weightRoot(sh, w) {
  const parts = [ascii(`${CBC_DOMAIN}root:`)];
  for (const name of [...TENSOR_NAMES].sort()) {
    const [lo, hi] = sh.offsets[name];
    const header = canonicalJson({ name, shape: sh.shapes[name], dtype: "<f8" });
    const leaf = toHex(sha256(concatBytes(ascii(`${CBC_DOMAIN}leaf:`), ascii(header), ascii(":"), f64Bytes(w.subarray(lo, hi)))));
    parts.push(ascii(`${name}:${leaf};`));
  }
  return toHex(sha256(concatBytes(...parts)));
}

export function normalisePins(pins, count) {
  const out = Array.from(new Set(Array.from(pins, (p) => p | 0))).sort((a, b) => a - b);
  if (out.length && (out[0] < 0 || out[out.length - 1] >= count)) throw new Error("pin outside the host's domain");
  return out;
}

export function claimId(kind, w0Root, w1Root, tau, pins) {
  const body = { schema: "heedle-claim/1", host: kind, w0_root: w0Root, w1_root: w1Root, tau: String(tau), pins: [...pins].sort((a, b) => a - b) };
  return toHex(sha256(ascii(`${DOMAIN}claim:${canonicalJson(body)}`)));
}

/** The seed of the probes drawn after a candidate is fixed (the probe order of CBC v1). */
export function probeSeed(w1Root, claim, testSet = "admission") {
  return sha256(ascii(`${DOMAIN}probes:${testSet}:${w1Root}:${claim}`));
}

/** The j-th 32-bit integer of the i-th hash of a probe seed. */
export function seededUint32(seed, i, j = 0) {
  const buf = new Uint8Array(36);
  buf.set(seed);
  new DataView(buf.buffer).setUint32(32, i, false);
  return new DataView(sha256(buf).buffer).getUint32(4 * j, false);
}

export function eventId(event) {
  const body = {};
  for (const k of ["seq", "kind", "prev", "parent_root", "root", "move_sha256", "pins", "gesture", "tau"]) {
    if (k in event) body[k] = event[k];
  }
  return toHex(sha256(ascii(`${DOMAIN}event:${canonicalJson(body)}`)));
}

const B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

function bytesToB64(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i += 3) {
    const a = bytes[i], b = bytes[i + 1] ?? 0, c = bytes[i + 2] ?? 0;
    s += B64[a >> 2] + B64[((a & 3) << 4) | (b >> 4)];
    s += i + 1 < bytes.length ? B64[((b & 15) << 2) | (c >> 6)] : "=";
    s += i + 2 < bytes.length ? B64[c & 63] : "=";
  }
  return s;
}

export function moveToB64(d) {
  return bytesToB64(f64Bytes(d));
}

/**
 * Autonomy zones, with the thresholds of the governor layer in the corpus's genesis dialogue:
 * tension below 0.3 the loom may weave on release, from 0.3 a deliberate hold is needed, and
 * from 0.7 a long one (the weaver's knot). Tension is a prediction, so it never refuses by
 * itself: weft's gate refuses, and a refuted claim snaps.
 */
export const ZONES = ["auto", "shared", "knot"];
export const GESTURES = ["release", "hold", "knot"];

export function zoneOf(tension) {
  if (tension < 0.3) return "auto";
  if (tension < 0.7) return "shared";
  return "knot";
}

/** Whether a gesture is deliberate enough for a zone. */
export function gestureSuffices(gesture, zone) {
  return GESTURES.indexOf(gesture) >= ZONES.indexOf(zone);
}

/**
 * One steering session at a host. The loom holds the baseline, the lineage of fells, the
 * pins, and the events of the draft. Every pick goes through weft's gate against the
 * baseline, is rooted, and carries a claim that the host's dispute tries to refute.
 */
export class Loom {
  constructor({ host, shafts = 16 }) {
    this.host = host;
    this.sh = host;                       // a host carries its own shape
    this.rInvariant = host.rInvariant;
    this.cost = host.cost;
    this.tau = String(host.tau);
    this.tauValue = Number(host.tau);
    this.shafts = shafts;
    this.base = host.w0;
    this.baseRoot = weightRoot(host, this.base);
    this.initialBudget = host.budget;
    this.lineage = [{ root: this.baseRoot, w: this.base, budget: host.budget }];
    this.events = [];
    this.pins = [];
    this.fitInputs = host.fitInputs();
    this.fit = host.encode(this.fitInputs);
    this.threading = null;
    this.shed = null;
    this.rethread();
  }

  get fell() {
    return this.lineage[this.lineage.length - 1];
  }

  /** Linearise at the fell, around the raised heddles: shed, shafts, treadle patterns, tie-up. */
  rethread({ iterations } = {}) {
    const w = this.fell.w, host = this.host;
    this.shed = shedOf(host, w, this.pins.length ? host.encode(host.pinInputs(this.pins)) : null);
    const warm = this.threading && this.threading.V.length === this.shafts ? this.threading : null;
    this.threading = thread(host, w, this.fit, this.shafts, { iterations: iterations ?? (warm ? 8 : 32), warm, basis: this.shed.Q });
    const y = forward(host, w, this.fit);
    this.outputs = y;
    const patterns = {};
    for (const name of host.treadleNames) patterns[name] = host.treadlePattern(name, this.fitInputs, y);
    this.tie = tieUp(this.threading, patterns);
    return this.threading;
  }

  setShafts(k) {
    this.shafts = k;
    this.threading = null;
    this.rethread();
  }

  /** Raise or lower heddles. The loom is re-threaded around them. */
  setPins(pins) {
    this.pins = normalisePins(pins, this.host.pinCount);
    this.rethread({ iterations: 16 });
  }

  /** The move for treadle travels and shaft travels, passed through the shed. */
  moveFor(treadling = {}, shaftTravel = {}) {
    const P = this.host.P;
    const d = new Float64Array(P);
    for (const [name, t] of Object.entries(treadling)) {
      if (!t) continue;
      const u = this.tie.unitMove[name];
      for (let i = 0; i < P; i++) d[i] += t * u[i];
    }
    for (const [i, t] of Object.entries(shaftTravel)) {
      if (!t) continue;
      const s = t / Math.sqrt(this.threading.lam[i]);
      const v = this.threading.V[i];
      for (let k = 0; k < P; k++) d[k] += s * v[k];
    }
    return this.shed.project(d);
  }

  /** Everything the weaver sees before beating a move in. Nothing is committed. */
  assess(d) {
    const fell = this.fell;
    const cand = addScaled(fell.w, d, 1.0);
    const gate = verifyWeightUpdateSafety(this.base, cand, this.rInvariant, fell.budget, this.cost);
    const first = firstOrder(this.threading, d);
    const predicted = Math.sqrt(dot(first, first) / first.length);
    const predictedMax = worstRow(first, this.host.M);
    const ball = gate.distance / this.rInvariant;
    const behaviour = predictedMax / this.tauValue;
    const tension = Math.max(Number.isFinite(ball) ? ball : Infinity, behaviour);
    return { cand, gate, first, predicted, predictedMax, ball, behaviour, tension, zone: gate.accepted ? zoneOf(tension) : "refused", length: norm(d) };
  }

  /**
   * Beat a move in. `gesture` records how the weaver committed: "release", "hold" or "knot".
   * `treadling` and `note` say what was pressed; they are kept as readings, not hashed.
   *
   *   { refused }  weft's gate said no. Nothing is recorded and no budget is spent.
   *   { needs }    the gesture was not deliberate enough for the pick's zone. Nothing happens.
   *   event        kind "pick": the gate accepted, and the host's dispute found no input that
   *                changes by more than tau. The move becomes the fell.
   *                kind "snap": the gate accepted, and one input refutes the claim. The fell
   *                does not move; the witness is kept as the reason; the budget is spent.
   */
  commit(d, gesture = "hold", treadling = {}, note = "") {
    const fell = this.fell, host = this.host;
    const a = this.assess(d);
    if (!a.gate.accepted) return { refused: a.gate };
    if (!gestureSuffices(gesture, a.zone)) return { needs: a.zone };
    const root = weightRoot(host, a.cand);
    const moveBytes = f64Bytes(d);
    const cid = claimId(host.kind, fell.root, root, this.tau, this.pins);
    const probeInputs = host.freshInputs(probeSeed(root, cid), host.probes);
    const probes = host.encode(probeInputs);
    const y0 = forward(host, fell.w, probes), y1 = forward(host, a.cand, probes);
    const dy = y1.map((v, i) => v - y0[i]);
    const realized = Math.sqrt(dot(dy, dy) / dy.length);
    const dispute = host.dispute(fell.w, a.cand);
    const refuted = dispute.worst > this.tauValue;
    let pinLeak = 0.0;
    if (this.pins.length) for (const c of change(host, fell.w, a.cand, host.encode(host.pinInputs(this.pins)))) pinLeak = Math.max(pinLeak, c);
    // `promised` is the first-order gain of this very move on the fit probes (so it already
    // counts the pins and the other treadles); `gain` is what the fresh probes measured.
    const delivered = {};
    for (const name of Object.keys(treadling).filter((k) => treadling[k])) {
      delivered[name] = {
        asked: treadling[name],
        promised: gainAlong(a.first, this.tie.patterns[name]),
        gain: gainAlong(dy, host.treadlePattern(name, probeInputs, y0)),
      };
    }
    const event = {
      seq: this.events.length + 1,
      kind: refuted ? "snap" : "pick",
      prev: this.events.length ? this.events[this.events.length - 1].id : "genesis",
      parent_root: fell.root,
      root,
      move_sha256: toHex(sha256(moveBytes)),
      move_b64: bytesToB64(moveBytes),
      pins: [...this.pins],
      gesture,
      tau: this.tau,
      gate: { accepted: true, distance: a.gate.distance, remaining_budget: a.gate.remainingBudget, reason: a.gate.reason },
      reading: { length: a.length, predicted: a.predicted, predicted_worst: a.predictedMax, tension: a.tension, zone: a.zone, pin_leak: pinLeak, treadling, delivered, note },
      claim: { id: cid, probes: host.probes, realized, worst: dispute.worst, witness: dispute.witness, rows: dispute.rows, exhaustive: dispute.exhaustive, refuted },
    };
    event.id = eventId(event);
    this.events.push(event);
    if (refuted) {
      // The thread snaps: the weights stay where they were, the attempt stays on the record.
      this.lineage.push({ root: fell.root, w: fell.w, budget: a.gate.remainingBudget });
    } else {
      this.lineage.push({ root, w: a.cand, budget: a.gate.remainingBudget });
      this.rethread();
    }
    return event;
  }

  /** Return to an earlier fell. The picks stay in the draft; the budget spent stays spent. */
  unpick(root) {
    let at = -1;
    for (let i = this.lineage.length - 1; i >= 0; i--) if (this.lineage[i].root === root) { at = i; break; }
    if (at < 0) throw new Error("that root was never the fell");
    const fell = this.fell;
    const event = {
      seq: this.events.length + 1,
      kind: "unpick",
      prev: this.events.length ? this.events[this.events.length - 1].id : "genesis",
      parent_root: fell.root,
      root,
      pins: [...this.pins],
      gesture: "hold",
      tau: this.tau,
    };
    event.id = eventId(event);
    this.events.push(event);
    this.lineage.push({ root, w: this.lineage[at].w, budget: fell.budget });
    this.rethread({ iterations: 24 });
    return event;
  }

  /** Which picks are in the cloth now, and the roots along the way: the draft, walked as a replay walks it. */
  path() {
    const roots = [this.baseRoot], picks = [];
    for (const ev of this.events) {
      if (ev.kind === "pick") {
        roots.push(ev.root);
        picks.push(ev);
      } else if (ev.kind === "unpick") {
        const at = roots.lastIndexOf(ev.root);
        roots.length = at + 1;
        picks.length = at;
      }
    }
    return { roots, picks };
  }

  /** The draft: enough to rebuild every fell and re-check every verdict without this page. */
  exportDraft() {
    return {
      schema: "heedle-draft/1",
      host: this.host.descriptor(),
      gate: { r_invariant: this.rInvariant, budget: this.initialBudget, cost: this.cost, tau: this.tau },
      events: this.events,
    };
  }
}
