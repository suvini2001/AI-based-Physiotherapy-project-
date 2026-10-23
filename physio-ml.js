/**
 * physio-ml.js — in-browser learned elbow-angle correction.
 *
 * Two independent halves, deliberately separable so each can be tested alone:
 *
 *   LgbmModel     loads elbow_correction_compact.json and walks the trees.
 *                 An exact mirror of LightGBM's Tree::NumericalDecision,
 *                 including the three missing-value policies. Verified in
 *                 Python at leaf-index level: 1.000000 agreement over
 *                 240,000 (frame, tree) routings.
 *
 *   FeatureStream builds the 16-feature vector from MediaPipe output, port of
 *                 features.py + angles_lift.py + angles.py.
 *
 * FEATURE ORDER IS LOAD-BEARING. It must match KEEP_NAMES exactly:
 *   0 angle_2d    1 angle_world  2 angle_lift  3 plan_upper  4 plan_fore
 *   5 plan_min    6 len_upper    7 len_fore    8 vis_sh      9 vis_el
 *  10 vis_wr     11 wz_sh_el    12 wz_el_wr   13 torso_ratio 14 d_angle
 *  15 std_angle
 * The model checks this against the names in the JSON on load and throws if
 * they disagree, because a silent re-ordering would produce plausible wrong
 * angles rather than an error.
 *
 * WHY THERE IS A LAG. Three of the Python features use a CENTRED window:
 * rolling std of the angle (15), rolling median of the depth term (5) and a
 * rolling majority sign (9). A causal trailing window is not the same
 * function, and the model was trained on the centred one. So this module
 * emits the vector for frame (t - LAG) with LAG = 7, which is exactly the
 * padding the widest window needs. At 30 fps that is 233 ms of delay — real,
 * but the alternative is feeding the model features it has never seen.
 *
 * No dependencies. ES module. Companion to physio-core.js.
 */

// ----------------------------------------------------------------- constants
export const FEATURE_NAMES = [
  'angle_2d', 'angle_world', 'angle_lift',
  'plan_upper', 'plan_fore', 'plan_min',
  'len_upper', 'len_fore',
  'vis_sh', 'vis_el', 'vis_wr',
  'wz_sh_el', 'wz_el_wr',
  'torso_ratio', 'd_angle', 'std_angle',
];

const L_SH = 11, R_SH = 12, L_HIP = 23, R_HIP = 24;
const ARM = { right: [12, 14, 16], left: [11, 13, 15] };   // shoulder, elbow, wrist

/** angles_lift.py / features.py tuning, unchanged. */
export const ML_DEFAULTS = {
  visibilityThreshold: 0.70,
  calibFrames: 90,        // 3 s at 30 fps before the scale is trusted
  minBoneFrames: 30,      // bone_lengths min_frames
  bonePercentile: 95,     // bone_lengths pct
  fullPlane: 0.95,        // lift_segments full_plane
  minRatio: 0.30,         // elbow_angle_lifted min_ratio
  depthSmooth: 5,         // lift_segments depth_smooth
  signWindow: 9,          // lift_segments sign_window
  stdWindow: 15,          // features.py _rolling_std
  minBoneLen: 0.10,       // angles.py extract_ordered_arm_joints
  maxBoneLen: 0.55,
  swapJump: 0.15,
  swapNear: 0.05,
};

export const LAG = 7;     // (max(stdWindow, signWindow, depthSmooth) - 1) / 2

// -------------------------------------------------------------------- model
const MISSING_NONE = 0, MISSING_ZERO = 1, MISSING_NAN = 2;

export class LgbmModel {
  /**
   * @param {object} json parsed elbow_correction_compact.json
   * @param {string[]} [expectedNames] throw unless the JSON matches this order
   */
  constructor(json, expectedNames = FEATURE_NAMES) {
    this.trees = json.trees;
    this.names = json.feature_names;
    this.clip = json.clip || [0, 180];
    this.zeroThreshold = json.zero_threshold ?? 1e-35;

    if (expectedNames) {
      const a = this.names.join('|'), b = expectedNames.join('|');
      if (a !== b) {
        throw new Error(
          `feature order mismatch.\n  model: ${a}\n  code : ${b}`);
      }
    }
  }

  static async load(url, expectedNames = FEATURE_NAMES) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} loading ${url}`);
    return new LgbmModel(await res.json(), expectedNames);
  }

  /** Leaf index one tree sends this row to. Mirrors Tree::NumericalDecision. */
  routeOne(tree, row) {
    let nd = tree.root;
    while (nd >= 0) {
      let x = row[tree.f[nd]];
      const mt = tree.m[nd];
      let go = null;
      if (mt === MISSING_NAN) {
        if (!Number.isFinite(x)) go = tree.d[nd] ? tree.l[nd] : tree.r[nd];
      } else {
        if (!Number.isFinite(x)) x = 0.0;
        if (mt === MISSING_ZERO &&
          x > -this.zeroThreshold && x <= this.zeroThreshold) {
          go = tree.d[nd] ? tree.l[nd] : tree.r[nd];
        }
      }
      if (go === null) go = x <= tree.t[nd] ? tree.l[nd] : tree.r[nd];
      nd = go;
    }
    return -nd - 1;
  }

  /**
   * @param {number[]|Float64Array} row one feature vector, FEATURE_NAMES order
   * @returns {number} corrected angle in degrees
   */
  predict(row) {
    let sum = 0;
    for (const tree of this.trees) sum += tree.v[this.routeOne(tree, row)];
    return Math.min(this.clip[1], Math.max(this.clip[0], sum));
  }

  /** Leaf indices per tree — for comparing against Python's pred_leaf. */
  leaves(row) {
    return this.trees.map((t) => this.routeOne(t, row));
  }
}

// ------------------------------------------------------------------ helpers
function nanMedian(values) {
  const s = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!s.length) return NaN;
  const h = s.length >> 1;
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
}

/** numpy.percentile default ('linear') on finite values only. */
function percentile(values, pct) {
  const s = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!s.length) return NaN;
  const pos = ((s.length - 1) * pct) / 100;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return lo === hi ? s[lo] : s[lo] + (pos - lo) * (s[hi] - s[lo]);
}

/** numpy.nanstd, ddof = 0. NaN if the window is entirely NaN. */
function nanStd(values) {
  const v = values.filter(Number.isFinite);
  if (!v.length) return NaN;
  const m = v.reduce((a, b) => a + b, 0) / v.length;
  return Math.sqrt(v.reduce((a, b) => a + (b - m) * (b - m), 0) / v.length);
}

/**
 * A centred window over a buffer, with numpy's mode='edge' padding.
 * @param {Array} buf  oldest-first
 * @param {number} i   centre index into buf
 * @param {number} win window size, forced odd
 * @param {Function} get  buf element -> number
 */
function centredWindow(buf, i, win, get) {
  const k = win | 1, pad = k >> 1, out = [];
  for (let d = -pad; d <= pad; d++) {
    const j = Math.min(buf.length - 1, Math.max(0, i + d));
    out.push(get(buf[j]));
  }
  return out;
}

/** Angle at b between b->a and b->c. Port of angles.py angle(). */
function angleBetween(a, b, c) {
  if (!a || !b || !c) return NaN;
  const n = Math.min(a.length, b.length, c.length);
  let dot = 0, nu = 0, nv = 0;
  for (let i = 0; i < n; i++) {
    const u = a[i] - b[i], v = c[i] - b[i];
    if (!Number.isFinite(u) || !Number.isFinite(v)) return NaN;
    dot += u * v; nu += u * u; nv += v * v;
  }
  nu = Math.sqrt(nu); nv = Math.sqrt(nv);
  if (nu < 1e-9 || nv < 1e-9) return NaN;
  return (Math.acos(Math.min(1, Math.max(-1, dot / (nu * nv)))) * 180) / Math.PI;
}

function dist3(p, q) {
  return Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
}

// ----------------------------------------------------- session calibration
/**
 * Holds the two quantities features.py computes once per recording: the
 * session shoulder-width scale and the two bone lengths. In an offline
 * recording numpy sees every frame; live, these have to be estimated from a
 * calibration window and then FROZEN, or the features drift under the model
 * as the session goes on.
 *
 * bone_lengths' decoupling is preserved: the percentile is taken on the raw
 * PIXEL length and divided by the median scale afterwards. Taking it on a
 * per-frame ratio inflated the estimate by ~60% on this project's own clip,
 * because frames with foreshortened shoulders spike the ratio.
 */
export class SessionScale {
  constructor(opt = {}) {
    this.o = { ...ML_DEFAULTS, ...opt };
    this.sw = [];
    this.pxUpper = [];
    this.pxFore = [];
    this.frozen = null;
    this.cap = 6000;        // ~3 min at 30 fps, then decimate
  }

  _push(arr, v) {
    if (!Number.isFinite(v)) return;
    arr.push(v);
    if (arr.length > this.cap) {
      for (let i = 0, j = 0; i < arr.length; i += 2, j++) arr[j] = arr[i];
      arr.length = Math.ceil(this.cap / 2);
    }
  }

  observe({ sw, pxUpper, pxFore }) {
    if (this.frozen) return;
    this._push(this.sw, sw);
    this._push(this.pxUpper, pxUpper);
    this._push(this.pxFore, pxFore);
  }

  get ready() {
    return this.frozen != null ||
      (this.sw.length >= this.o.calibFrames &&
        this.pxUpper.length >= this.o.minBoneFrames &&
        this.pxFore.length >= this.o.minBoneFrames);
  }

  /** Compute and freeze. Call once `ready`; idempotent. */
  freeze() {
    if (this.frozen) return this.frozen;
    const sc = nanMedian(this.sw);
    if (!(sc > 0)) return null;
    const p = this.o.bonePercentile;
    this.frozen = {
      scale: sc,
      lUpper: this.pxUpper.length >= this.o.minBoneFrames
        ? percentile(this.pxUpper, p) / sc : NaN,
      lFore: this.pxFore.length >= this.o.minBoneFrames
        ? percentile(this.pxFore, p) / sc : NaN,
      frames: this.sw.length,
    };
    return this.frozen;
  }

  recalibrate() {
    this.frozen = null;
    this.sw.length = 0; this.pxUpper.length = 0; this.pxFore.length = 0;
  }

  /**
   * Force the calibration instead of estimating it. Two uses: replaying a
   * known user's saved calibration, and verifying this port against Python,
   * where numpy had the whole clip and this stream only has a window.
   * @param {{scale:number, lUpper:number, lFore:number}} cal
   */
  set(cal) {
    this.frozen = { frames: 0, ...cal };
    return this.frozen;
  }
}

// -------------------------------------------------------------- the stream
/**
 * Push one frame of MediaPipe output per video frame; get back the feature
 * vector for the frame LAG frames ago, or null while still filling.
 *
 * @example
 *   const stream = new FeatureStream({ side: 'right', size: { w, h } });
 *   const model  = await LgbmModel.load('/elbow_correction_compact.json');
 *   // per frame:
 *   const out = stream.push(result.landmarks, result.worldLandmarks);
 *   if (out) corrected = model.predict(out.features);
 */
export class FeatureStream {
  /**
   * @param {object} opt
   * @param {'right'|'left'} opt.side
   * @param {{w:number,h:number}} opt.size video pixel size
   * @param {object} [opt.tuning] overrides for ML_DEFAULTS
   * @param {SessionScale} [opt.scale] share one across both arms if you like
   */
  constructor({ side, size, tuning = {}, scale = null }) {
    this.side = side;
    this.size = size;
    this.o = { ...ML_DEFAULTS, ...tuning };
    this.scale = scale || new SessionScale(this.o);
    this.buf = [];                       // oldest-first, length LAG*2+1 max
    this.keep = LAG * 2 + 1;
    this.prevElbow = null;               // for the anti-swap check
    this.emitted = 0;
  }

  /** visibility-masked point as [x*w, y*h], or null. Port of visibility.py. */
  _px(lm, i) {
    const p = lm[i];
    if (!p) return null;
    const vis = p.visibility ?? 1;
    if (vis < this.o.visibilityThreshold) return null;
    if (p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1) return null;
    return [p.x * this.size.w, p.y * this.size.h];
  }

  _vis(lm, i) {
    const p = lm[i];
    if (!p) return NaN;
    const vis = p.visibility ?? 1;
    if (vis < this.o.visibilityThreshold) return NaN;
    if (p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1) return NaN;
    return vis;
  }

  /** Ordered [shoulder, elbow, wrist] world points with anti-swap and the
   *  anatomical bone gate. Port of angles.py extract_ordered_arm_joints. */
  _orderedWorld(world, lm) {
    const [sh, el, wr] = ARM[this.side];
    const opp = ARM[this.side === 'right' ? 'left' : 'right'];
    const get = (i) => {
      const p = world[i];
      if (!p) return null;
      if (this._vis(lm, i) !== this._vis(lm, i)) return null;   // masked -> NaN
      const v = [p.x, p.y, p.z];
      return v.every(Number.isFinite) ? v : null;
    };
    let pts = [get(sh), get(el), get(wr)];

    // anti-swap: the elbow jumped far AND landed on the other arm's elbow
    const oppEl = get(opp[1]);
    if (this.prevElbow && pts[1] && oppEl) {
      if (dist3(pts[1], this.prevElbow) > this.o.swapJump &&
        dist3(pts[1], oppEl) < this.o.swapNear) {
        pts = [null, null, null];
      }
    }
    this.prevElbow = pts[1];

    // anatomical gate: both bones between minBoneLen and maxBoneLen metres
    if (pts[0] && pts[1] && pts[2]) {
      const lu = dist3(pts[0], pts[1]), lf = dist3(pts[2], pts[1]);
      const bad = lu < this.o.minBoneLen || lu > this.o.maxBoneLen ||
        lf < this.o.minBoneLen || lf > this.o.maxBoneLen;
      if (bad) pts = [null, null, null];
    }
    return pts;
  }

  /**
   * @param {Array} landmarks       33 normalised landmarks
   * @param {Array} worldLandmarks  33 world landmarks (metres), may be null
   * @returns {{features:number[], index:number, status:string}|null}
   */
  push(landmarks, worldLandmarks) {
    const [sh, el, wr] = ARM[this.side];
    const lm = landmarks || [];
    const wd = worldLandmarks || [];

    const pSh = this._px(lm, sh), pEl = this._px(lm, el), pWr = this._px(lm, wr);
    const pLS = this._px(lm, L_SH), pRS = this._px(lm, R_SH);
    const pLH = this._px(lm, L_HIP), pRH = this._px(lm, R_HIP);

    const swRaw = pLS && pRS
      ? Math.hypot(pLS[0] - pRS[0], pLS[1] - pRS[1]) : NaN;
    const sw = swRaw > 1e-6 ? swRaw : NaN;

    const dxU = pSh && pEl ? pSh[0] - pEl[0] : NaN;
    const dyU = pSh && pEl ? pSh[1] - pEl[1] : NaN;
    const dxF = pWr && pEl ? pWr[0] - pEl[0] : NaN;
    const dyF = pWr && pEl ? pWr[1] - pEl[1] : NaN;

    const ordered = this._orderedWorld(wd, lm);

    // torso_ratio uses the PER-FRAME shoulder width, not the session median
    let torsoRatio = NaN;
    if (pLS && pRS && pLH && pRH) {
      const midShY = (pLS[1] + pRS[1]) / 2;
      const midHipY = (pLH[1] + pRH[1]) / 2;
      const th = Math.abs(midHipY - midShY);
      if (th > 1e-6) torsoRatio = sw / th;
    }

    const wz = (i) => {
      const p = wd[i];
      return p && Number.isFinite(p.z) && Number.isFinite(this._vis(lm, i))
        ? p.z : NaN;
    };
    const wzSh = wz(sh), wzEl = wz(el), wzWr = wz(wr);

    this.scale.observe({
      sw,
      pxUpper: Math.hypot(dxU, dyU),
      pxFore: Math.hypot(dxF, dyF),
    });

    this.buf.push({
      dxU, dyU, dxF, dyF, sw, torsoRatio,
      angle2d: angleBetween(pSh, pEl, pWr),
      angleWorld: angleBetween(ordered[0], ordered[1], ordered[2]),
      visSh: this._vis(lm, sh), visEl: this._vis(lm, el), visWr: this._vis(lm, wr),
      wzSh, wzEl, wzWr,
      // the model decides only front/back; sign of the world depth difference
      signU: Number.isFinite(wzSh - wzEl) ? Math.sign(wzSh - wzEl) || 1 : 1,
      signF: Number.isFinite(wzWr - wzEl) ? Math.sign(wzWr - wzEl) || 1 : 1,
    });
    if (this.buf.length > this.keep) this.buf.shift();

    if (!this.scale.ready) {
      return null;                                 // still calibrating
    }
    const cal = this.scale.freeze();
    if (!cal) return null;

    // the frame we can still centre a window on
    const i = this.buf.length - 1 - LAG;
    if (i < 0) return null;

    return {
      features: this._vectorAt(i, cal),
      index: this.emitted++,
      status: 'ok',
      calibration: cal,
    };
  }

  /** Build the 16-vector for buf[i]. All windows centred on i. */
  _vectorAt(i, cal) {
    const f = this.buf[i];
    const sc = cal.scale;
    const { lUpper, lFore } = cal;

    // --- lengths and planarity, shoulder-width units ---
    const lenUpper = Math.hypot(f.dxU, f.dyU) / sc;
    const lenFore = Math.hypot(f.dxF, f.dyF) / sc;
    const planUpper = lUpper > 0 ? lenUpper / lUpper : NaN;
    const planFore = lFore > 0 ? lenFore / lFore : NaN;
    const planMin = Math.min(planUpper, planFore);

    // --- the 2.5D lift, port of lift_segments + elbow_angle_lifted ---
    //
    // Order matters and is easy to get wrong: lift_segments applies the
    // in-plane deadband to the WHOLE series first, and only then takes the
    // rolling median. Applying the deadband at the centre frame and skipping
    // the median there is a different function — it cost 14.8 deg in the
    // port test. _rolling_median also restores NaN wherever the CENTRE
    // sample was NaN, so a NaN frame cannot be filled in by its neighbours.
    const depthAt = (g, L, upper) => {
      const dx = upper ? g.dxU : g.dxF;
      const dy = upper ? g.dyU : g.dyF;
      const planar = Math.hypot(dx, dy) / sc;
      if (!Number.isFinite(planar)) return NaN;
      const dz = Math.sqrt(Math.max(0, L * L - planar * planar));
      return planar / L > this.o.fullPlane ? 0 : dz;      // deadband first
    };

    const seg = (dx, dy, L, upper, signKey) => {
      if (!(L > 0)) return null;
      const x = dx / sc, y = dy / sc;
      const planar = Math.hypot(x, y);
      if (!Number.isFinite(planar)) return null;
      const ratio = planar / L;

      const centre = depthAt(f, L, upper);
      let dz = nanMedian(centredWindow(this.buf, i, this.o.depthSmooth,
        (g) => depthAt(g, L, upper)));
      if (!Number.isFinite(centre)) dz = NaN;   // NaN in -> NaN out

      // the model decides only which way; majority sign over a centred window
      const signs = centredWindow(this.buf, i, this.o.signWindow,
        (g) => g[signKey]);
      const acc = signs.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0);
      const s = Math.sign(acc) || 1;
      return { v: [x, y, dz * s], ratio };
    };

    const u = seg(f.dxU, f.dyU, lUpper, true, 'signU');
    const v = seg(f.dxF, f.dyF, lFore, false, 'signF');
    let angleLift = NaN;
    if (u && v) {
      const ratio = Math.min(u.ratio, v.ratio);
      if (Number.isFinite(ratio) && ratio >= this.o.minRatio) {
        angleLift = angleBetween(u.v, [0, 0, 0], v.v);
      }
    }

    // --- temporal features ---
    const prev = i > 0 ? this.buf[i - 1].angle2d : f.angle2d;
    const dAngle = i > 0 ? f.angle2d - prev : 0.0;
    const stdAngle = nanStd(
      centredWindow(this.buf, i, this.o.stdWindow, (g) => g.angle2d));

    return [
      f.angle2d,
      f.angleWorld,
      angleLift,
      planUpper,
      planFore,
      planMin,
      lenUpper,
      lenFore,
      f.visSh, f.visEl, f.visWr,
      f.wzSh - f.wzEl,
      f.wzEl - f.wzWr,
      f.torsoRatio,
      dAngle,
      stdAngle,
    ];
  }

  reset() {
    this.buf.length = 0;
    this.prevElbow = null;
    this.emitted = 0;
  }
}

// ------------------------------------------------------------ verification
/**
 * Check this file against the Python pipeline. Feed it the fixture written by
 * the make_js_fixture cell: the model's own predictions on a sample of real
 * feature vectors.
 *
 * This tests the MODEL WALKER only, which is the half that must be bit-exact.
 * The feature builder is checked separately by comparing the vectors this
 * stream produces against `fixture.features` for the same recording.
 *
 * @param {LgbmModel} model
 * @param {{features:number[][], predictions:number[], leaves?:number[][]}} fx
 * @returns {{rows:number, maxAbsDiff:number, leafAgreement:number, ok:boolean}}
 */
export function verifyModel(model, fx) {
  let maxAbsDiff = 0, leafMatch = 0, leafTotal = 0;
  for (let i = 0; i < fx.features.length; i++) {
    const got = model.predict(fx.features[i]);
    maxAbsDiff = Math.max(maxAbsDiff, Math.abs(got - fx.predictions[i]));
    if (fx.leaves) {
      const mine = model.leaves(fx.features[i]);
      for (let j = 0; j < mine.length; j++) {
        leafTotal++;
        if (mine[j] === fx.leaves[i][j]) leafMatch++;
      }
    }
  }
  const leafAgreement = leafTotal ? leafMatch / leafTotal : NaN;
  return {
    rows: fx.features.length,
    maxAbsDiff,
    leafAgreement,
    ok: maxAbsDiff < 1e-6 && (!fx.leaves || leafAgreement === 1),
  };
}