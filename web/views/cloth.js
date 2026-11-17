/**
 * The cloth view: a host whose inputs are points of the unit square, drawn as a woven cloth.
 * Every pixel is an input. The picture samples the square, 64 or 128 points a side.
 *
 * Colours are a fixed map of the three outputs, for the eye only. No check reads a colour.
 */
import * as hd from "../heedle.js";
import { grid, PIN_LATTICE } from "../hosts/cloth.js";

const LIVE = 64, SHARP = 128, THUMB = 24, SHAFT_PIC = 10;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// The three outputs are rotated into lightness, warm-cool and green-magenta, then OKLab.
const GAMMA_LUT = new Uint8ClampedArray(4097);
for (let i = 0; i <= 4096; i++) {
  const c = i / 4096;
  GAMMA_LUT[i] = Math.round(255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055));
}
const lut = (c) => GAMMA_LUT[c <= 0 ? 0 : c >= 1 ? 4096 : (c * 4096) | 0];

function paint(y, img) {
  const data = img.data;
  for (let i = 0; i < y.length / 3; i++) {
    const y0 = y[3 * i], y1 = y[3 * i + 1], y2 = y[3 * i + 2];
    const L = 0.6 + 0.25 * Math.tanh(0.5 * (y0 + y1 + y2) * 0.5773502691896258);
    const c1 = 0.115 * Math.tanh(0.6 * (y0 - y2) * 0.7071067811865476);
    const c2 = 0.1 * Math.tanh(0.6 * (2 * y1 - y0 - y2) * 0.408248290463863);
    const a = 0.55 * c1 - 0.83 * c2, b = 0.83 * c1 + 0.55 * c2;
    const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
    const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
    const s_ = L - 0.0894841775 * a - 1.291485548 * b;
    const l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    data[4 * i] = lut(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s);
    data[4 * i + 1] = lut(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s);
    data[4 * i + 2] = lut(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s);
    data[4 * i + 3] = 255;
  }
}

export function clothView() {
  let canvas, ctx, weave = null, fields = null, fellY = null;
  const veil = document.createElement("canvas");
  veil.width = veil.height = LIVE;
  const veilImg = new ImageData(LIVE, LIVE);

  const field = (host, n) => {
    const c = document.createElement("canvas");
    c.width = c.height = n;
    return { n, enc: host.encode(grid(n)), y: new Float64Array(n * n * 3), img: new ImageData(n, n), canvas: c, ctx: c.getContext("2d") };
  };
  const render = (host, f, w) => {
    hd.forward(host, w, f.enc, f.y);
    paint(f.y, f.img);
    f.ctx.putImageData(f.img, 0, 0);
  };

  /** A plain-weave relief: each cell of the 64 x 64 picture is one crossing of warp and weft. */
  const weavePattern = (cell) => {
    const t = document.createElement("canvas");
    t.width = t.height = Math.max(2, Math.round(cell * 2));
    const c = t.getContext("2d"), s = t.width / 2;
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        const g = (i + j) % 2 === 0 ? c.createLinearGradient(0, j * s, 0, (j + 1) * s) : c.createLinearGradient(i * s, 0, (i + 1) * s, 0);
        g.addColorStop(0, "rgba(0,0,0,0.26)");
        g.addColorStop(0.5, "rgba(255,255,255,0.10)");
        g.addColorStop(1, "rgba(0,0,0,0.26)");
        c.fillStyle = g;
        c.fillRect(i * s, j * s, s, s);
      }
    }
    return ctx.createPattern(t, "repeat");
  };

  function drawCloth(s, f) {
    const { loom, host } = s;
    const side = Math.round(canvas.clientWidth * Math.min(2, window.devicePixelRatio || 1));
    if (side > 0 && canvas.width !== side) {
      canvas.width = canvas.height = side;
      weave = null;
    }
    const S = canvas.width;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(f.canvas, 0, 0, S, S);
    weave ??= weavePattern(S / LIVE);
    ctx.fillStyle = weave;
    ctx.fillRect(0, 0, S, S);
    const ink = css("--ink"), paper = css("--paper");
    // Focus: dim what the pending move leaves alone, so the eye goes where the change will land.
    if (s.focus && !s.resting) {
      const live = fields.live, ch = new Float64Array(LIVE * LIVE);
      let top = 0;
      for (let i = 0; i < ch.length; i++) {
        const a = live.y[3 * i] - fellY[3 * i], b = live.y[3 * i + 1] - fellY[3 * i + 1], c = live.y[3 * i + 2] - fellY[3 * i + 2];
        ch[i] = Math.sqrt((a * a + b * b + c * c) / 3);
        if (ch[i] > top) top = ch[i];
      }
      if (top > 1e-4) {
        const rgb = paper.startsWith("#") ? [1, 3, 5].map((k) => parseInt(paper.slice(k, k + 2), 16)) : [128, 128, 128];
        for (let i = 0; i < ch.length; i++) {
          const quiet = 1 - Math.min(1, ch[i] / top);
          veilImg.data.set([rgb[0], rgb[1], rgb[2], Math.round(175 * quiet * quiet)], 4 * i);
        }
        veil.getContext("2d").putImageData(veilImg, 0, 0);
        ctx.drawImage(veil, 0, 0, S, S);
      }
    }
    const px = S / 640;
    const dot = (u, v, r, fill, stroke) => {
      ctx.beginPath();
      ctx.arc(u * S, v * S, r * px, 0, 2 * Math.PI);
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.lineWidth = 2 * px; ctx.strokeStyle = stroke; ctx.stroke(); }
    };
    if (s.probes) {
      for (let i = 0; i < loom.fitInputs.length; i += 2) dot(loom.fitInputs[i], loom.fitInputs[i + 1], 2.2, ink);
      if (s.lastEvent?.claim) {
        const fresh = host.freshInputs(hd.probeSeed(s.lastEvent.root, s.lastEvent.claim.id), s.lastEvent.claim.probes);
        ctx.strokeStyle = ink;
        ctx.lineWidth = 1.5 * px;
        for (let i = 0; i < fresh.length; i += 2) {
          const x = fresh[i] * S, y = fresh[i + 1] * S, r = 4 * px;
          ctx.beginPath();
          ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x - r, y + r); ctx.lineTo(x + r, y - r);
          ctx.stroke();
        }
      }
    }
    if (s.hidden) {
      const known = host.knownInputs();
      ctx.fillStyle = ink;
      for (let i = 0; i < known.length; i += 2) ctx.fillRect(known[i] * S - 4 * px, known[i + 1] * S - 4 * px, 8 * px, 8 * px);
    }
    const pins = host.pinInputs(loom.pins);
    for (let i = 0; i < pins.length; i += 2) {
      dot(pins[i], pins[i + 1], 7, paper, ink);
      dot(pins[i], pins[i + 1], 2, ink);
    }
    if (s.lastEvent?.claim && s.resting) {
      const [u, v] = s.lastEvent.claim.witness;
      dot(u, v, 11, null, css(s.lastEvent.kind === "snap" ? "--madder" : "--woad"));
    }
  }

  return {
    name: "cloth",
    inputWord: "point",
    provedText: "at 4,096 points of the cloth",
    knownText: " (the squares)",
    worded: false,
    futureLines: false,

    mount(slot, api) {
      canvas = document.createElement("canvas");
      canvas.id = "cloth";
      canvas.width = canvas.height = 640;
      canvas.setAttribute("role", "img");
      canvas.setAttribute("aria-label", "The cloth: the model's output at every point of the unit square. Click to pin a point.");
      slot.append(canvas);
      ctx = canvas.getContext("2d");
      weave = null;
      fields = null;
      canvas.addEventListener("click", (e) => {
        const loom = api.loom(), box = canvas.getBoundingClientRect();
        const at = (v, size) => Math.min(PIN_LATTICE - 1, Math.max(0, Math.floor((v / size) * PIN_LATTICE)));
        const col = at(e.clientX - box.left, box.width), row = at(e.clientY - box.top, box.height);
        // A click near a raised heddle lowers it.
        const near = loom.pins.find((p) => Math.abs((p % PIN_LATTICE) - col) <= 1 && Math.abs(Math.floor(p / PIN_LATTICE) - row) <= 1);
        api.setPins(near !== undefined ? loom.pins.filter((p) => p !== near) : [...loom.pins, row * PIN_LATTICE + col]);
      });
    },
    onFell(loom) {
      fields ??= { live: field(loom.host, LIVE), sharp: field(loom.host, SHARP), thumb: field(loom.host, THUMB) };
      fellY = hd.forward(loom.host, loom.fell.w, fields.live.enc);
    },
    draw(s) {
      render(s.host, fields.live, s.pending.a.cand);
      drawCloth(s, fields.live);
    },
    sharpen(s) {
      render(s.host, fields.sharp, s.pending.a.cand);
      drawCloth(s, fields.sharp);
    },
    shaftPicture(el, loom, i) {
      let c = el.querySelector("canvas");
      if (!c) {
        c = document.createElement("canvas");
        c.width = c.height = SHAFT_PIC;
        c.setAttribute("aria-hidden", "true");
        el.replaceChildren(c);
      }
      const pic = new ImageData(SHAFT_PIC, SHAFT_PIC), root = Math.sqrt(loom.threading.rows);
      paint(loom.threading.U[i].map((v) => 1.6 * v * root), pic);
      c.getContext("2d").putImageData(pic, 0, 0);
    },
    futureTile(button, loom, cand, label) {
      render(loom.host, fields.thumb, cand);
      button.innerHTML = `<canvas width="${THUMB}" height="${THUMB}" aria-hidden="true"></canvas><span>${label}</span>`;
      button.querySelector("canvas").getContext("2d").putImageData(fields.thumb.img, 0, 0);
    },
    witnessText: () => "it is ringed on the cloth",
  };
}
