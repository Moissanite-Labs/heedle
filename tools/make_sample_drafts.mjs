// Write drafts/sample-cloth.json and drafts/sample-text.json: one short scripted session at
// each loom, for the Python replay tests.   node tools/make_sample_drafts.mjs
//
// The moves are computed here, in JavaScript, and travel in the draft as float64 bytes, so
// the replay in Python reaches the same weight roots bit for bit. On another machine or
// engine the last digits of a move can differ; the draft that results is different and
// equally valid. The committed files are recorded sessions, not conformance vectors.
import fs from "node:fs";
import * as hd from "../web/heedle.js";
import { makeClothHost } from "../web/hosts/cloth.js";
import { makeTextHost } from "../web/hosts/text.js";

const repo = new URL("../", import.meta.url);
const write = (name, loom) => {
  fs.writeFileSync(new URL(`drafts/${name}.json`, repo), JSON.stringify(loom.exportDraft(), null, 1) + "\n");
  console.log(`wrote drafts/${name}.json: ${loom.events.map((e) => e.kind).join(", ")}`);
};
const presser = (loom) => (treadling, shafts = {}) => {
  const d = loom.moveFor(treadling, shafts);
  const zone = loom.assess(d).zone; // the lightest gesture the zone allows
  const result = loom.commit(d, { auto: "release", shared: "hold", knot: "knot" }[zone], treadling, "scripted");
  if (!result.kind) throw new Error(`not recorded: ${JSON.stringify(result)}`);
  return result;
};

{
  const loom = new hd.Loom({ host: makeClothHost(7) });
  const press = presser(loom);
  press({ warm: 0.08 });                      // a light touch
  loom.setPins([1056, 1057, 1120, 1121]);     // raise four heddles
  press({ tilt: 0.25 });
  const second = loom.fell.root;
  press({ contrast: 0.2, check: 0.2 });
  press({}, { 0: 0.45 });                     // too far along the stiffest shaft: this one snaps
  loom.unpick(second);
  loom.setPins([]);
  press({ light: -0.2 });
  write("sample-cloth", loom);
}
{
  const host = makeTextHost(JSON.parse(fs.readFileSync(new URL("hosts/text-v1.json", repo), "utf8")));
  const loom = new hd.Loom({ host });
  const press = presser(loom);
  press({ vowels: 0.15 });
  loom.setPins([host.contextOf(" th"), host.contextOf("the")]);   // "the" must stay as it is
  press({ spaces: 0.4 });
  const second = loom.fell.root;
  press({ echo: 0.5, sharp: -0.3 });
  press({}, { 0: 0.9 });                      // this one snaps, on a context the dispute names
  loom.unpick(second);
  loom.setPins([]);
  press({ stops: 0.3 });
  write("sample-text", loom);
}
