/**
 * The text view: a host whose inputs are contexts of three characters. Each pick weaves one
 * line of text from the new weights. Below the fell, the line a pending move would weave is
 * drawn as the knobs turn. Two sets of weights are always compared on the same random draws,
 * so a difference in the text is a difference in the model.
 *
 * Sampled text is for the eye only. No check reads a sampled character.
 */
import * as hd from "../heedle.js";
import { ALPHABET } from "../hosts/text.js";

const LINE = 56, OPENING = "the ", ROWS = 12;
const glyph = (ch) => (ch === " " ? "␣" : ch);
const esc = (ch) => (ch === " " ? "&nbsp;" : ch);

export function textView() {
  let woven, ahead, grid, api, lines = [], tail = OPENING, rowContexts = null, fellProb = null;

  const softmax = (y, o, V) => {
    let top = -Infinity, sum = 0.0;
    for (let k = 0; k < V; k++) top = Math.max(top, y[o + k]);
    const p = new Float64Array(V);
    for (let k = 0; k < V; k++) sum += p[k] = Math.exp(y[o + k] - top);
    for (let k = 0; k < V; k++) p[k] /= sum;
    return p;
  };
  const probs = (host, w) => {
    const y = hd.forward(host, w, host.encode(rowContexts));
    return Array.from(rowContexts, (_, i) => softmax(y, i * host.V, host.V));
  };

  function buildGrid(loom) {
    const host = loom.host;
    rowContexts = loom.fitInputs.slice(0, ROWS);
    const head = `<tr><th class="ctx">after</th>${[...ALPHABET].map((ch) => `<th>${glyph(ch)}</th>`).join("")}</tr>`;
    const rows = Array.from(rowContexts, (c, i) =>
      `<tr data-row="${i}"><td class="ctx" tabindex="0" role="button" aria-label="Pin what follows ${host.contextText(c)}">${[...host.contextText(c)].map(glyph).join("")}</td>${"<td><i></i></td>".repeat(host.V)}</tr>`).join("");
    grid.innerHTML = `<table>${head}${rows}</table>`;
    grid.querySelectorAll("td.ctx").forEach((cell) => {
      const toggle = () => {
        const c = rowContexts[Number(cell.parentElement.dataset.row)], pins = api.loom().pins;
        api.setPins(pins.includes(c) ? pins.filter((p) => p !== c) : [...pins, c]);
      };
      cell.addEventListener("click", toggle);
      cell.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), toggle()));
    });
  }

  return {
    name: "text",
    inputWord: "context",
    provedText: "at every context",
    knownText: ", its commonest contexts",
    worded: true,
    futureLines: true,

    mount(slot, api_) {
      api = api_;
      slot.innerHTML = `<div class="weave" aria-live="off"><div id="woven"></div><div class="fellmark"><span>the fell</span></div><div class="line ahead"><span class="ix">next</span><span class="txt" id="ahead"></span></div></div><div class="pattern" id="pattern"></div>`;
      woven = slot.querySelector("#woven");
      ahead = slot.querySelector("#ahead");
      grid = slot.querySelector("#pattern");
      rowContexts = null;
    },
    /** The cloth so far: one line per fell along the path, each continuing the last. */
    onFell(loom) {
      const host = loom.host;
      if (!rowContexts) buildGrid(loom);
      const weights = new Map(loom.lineage.map((l) => [l.root, l.w]));
      lines = [];
      tail = OPENING;
      loom.path().roots.forEach((root, i) => {
        const text = host.weave(weights.get(root), tail, LINE, i + 1).text;
        lines.push(text);
        tail = (tail + text).slice(-host.K);
      });
      woven.innerHTML = lines.map((text, i) => `<div class="line"><span class="ix">${i}</span><span class="txt">${i === 0 ? OPENING : ""}${text}</span></div>`).join("");
      fellProb = probs(host, loom.fell.w);
      grid.querySelectorAll("tr[data-row]").forEach((tr) => tr.classList.toggle("pinned", loom.pins.includes(rowContexts[Number(tr.dataset.row)])));
    },
    draw(s) {
      const { loom, host } = s, cand = s.pending.a.cand, seed = lines.length + 1;
      // The next line, at the fell and at the candidate, on the same draws.
      const now = host.weave(loom.fell.w, tail, LINE, seed);
      const next = s.resting ? now : host.weave(cand, tail, LINE, seed);
      let html = "";
      const ch = s.resting ? null : hd.change(host, loom.fell.w, cand, host.encode(next.contexts));
      let top = 0;
      if (ch) for (const v of ch) top = Math.max(top, v);
      for (let i = 0; i < LINE; i++) {
        // Focus: characters whose distribution the move leaves alone are dimmed.
        const quiet = ch && s.focus && top > 1e-6 ? 1 - Math.min(1, ch[i] / top) : 0;
        const differs = next.text[i] !== now.text[i];
        html += `<span${differs ? ' class="new"' : ""} style="opacity:${(1 - 0.72 * quiet * quiet).toFixed(2)}">${esc(next.text[i])}</span>`;
      }
      ahead.innerHTML = html;
      // The pattern: the model's guess after each of its commonest contexts.
      const p = s.resting ? fellProb : probs(host, cand);
      grid.querySelectorAll("tr[data-row]").forEach((tr) => {
        const i = Number(tr.dataset.row), cells = tr.children;
        for (let k = 0; k < host.V; k++) {
          const size = Math.round(13 * Math.sqrt(p[i][k]));
          const mark = cells[k + 1].firstChild, delta = p[i][k] - fellProb[i][k];
          mark.style.width = mark.style.height = `${size}px`;
          cells[k + 1].className = delta > 0.02 ? "up" : delta < -0.02 ? "down" : "";
          cells[k + 1].title = `${Math.round(100 * p[i][k])}%`;
        }
      });
    },
    /** What a shaft does, in characters: the ones it raises most and lowers most across the probes. */
    shaftPicture(el, loom, i) {
      const V = loom.host.V, u = loom.threading.U[i], mean = new Float64Array(V);
      for (let r = 0; r < u.length; r++) mean[r % V] += u[r];
      const order = Array.from(mean.keys()).sort((a, b) => mean[b] - mean[a]);
      const show = (list) => list.map((k) => glyph(ALPHABET[k])).join("");
      el.innerHTML = `<span class="up">+${show(order.slice(0, 3))}</span> <span class="down">−${show(order.slice(-3).reverse())}</span>`;
      el.title = "The characters this shaft raises most and lowers most, on average over the fit probes";
    },
    futureTile(button, loom, cand, label) {
      const text = loom.host.weave(cand, tail, LINE, lines.length + 1).text;
      button.innerHTML = `<span class="t">${label}</span><span class="txt"></span>`;
      button.lastChild.textContent = text;
    },
    witnessText: (loom, witness) => `the context “${loom.host.contextText(witness[0])}”`,
  };
}
