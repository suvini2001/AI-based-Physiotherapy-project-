/**
 * replayProducer.js — Person B, Week 1.
 * Replays recorded "sessions" as a stream of 20-byte live frames, so the live
 * channel and dashboard can be built without a real phone. Encodes exactly the
 * Live Frame Binary Layout v1 from Formats.md (little-endian, 20 bytes).
 *
 * Switches let you simulate a bad network:
 *   drop   — fraction of frames thrown away (0..1)
 *   pause  — ms to freeze the stream at a random point, mimicking a stall
 *   jitter — +/- ms randomness added to the 100ms (10 Hz) frame interval
 *
 * Usage (browser or Node):
 *   const p = new ReplayProducer(MOCK_SESSIONS[0], { drop: 0.1, jitter: 40 });
 *   p.onFrame = (buf) => sendOverWebSocket(buf);   // buf is a 20-byte ArrayBuffer
 *   p.start();  ...  p.pauseFor(2000);  ...  p.stop();
 */

export const STATE = { READY: 0, CALIB: 1, COUNT: 2 };
export const FAULT = { none: 0, elbow_too_low: 1, body_sway: 2, incomplete_rom: 3 };

/** Encode one frame to the 20-byte layout in Formats.md. */
export function encodeFrame({ timestamp, joint_angle, state, rep_count, fault_code }) {
  const buf = new ArrayBuffer(20);
  const dv = new DataView(buf);
  dv.setUint8(0, 1);                                   // version
  dv.setFloat32(1, timestamp, true);                   // little-endian
  dv.setFloat32(5, joint_angle, true);
  dv.setUint8(9, state & 0xff);
  dv.setUint16(10, rep_count & 0xffff, true);
  dv.setUint8(12, fault_code & 0xff);
  // bytes 13..19 reserved, already zero
  return buf;
}

/** A few synthetic sessions: a clean rep stream, one with a fault, one short. */
export function makeMockSessions() {
  const repWave = (n, peak, fault) => {
    const frames = [];
    for (let r = 0; r < n; r++) {
      for (let k = 0; k < 20; k++) {                   // 20 frames per rep ≈ 2s at 10Hz
        const angle = 40 + (peak - 40) * Math.sin((k / 20) * Math.PI); // 40→peak→40
        frames.push({
          joint_angle: Math.round(angle),
          state: STATE.COUNT,
          rep_count: r,
          fault_code: fault && r >= n - 2 ? fault : FAULT.none,
        });
      }
    }
    return frames;
  };
  return [
    { id: "clean", frames: repWave(10, 150, FAULT.none) },
    { id: "shoulder_hike", frames: repWave(10, 150, FAULT.body_sway) },
    { id: "shallow", frames: repWave(6, 110, FAULT.incomplete_rom) },
  ];
}
export const MOCK_SESSIONS = makeMockSessions();

export class ReplayProducer {
  constructor(session, opts = {}) {
    this.frames = session.frames;
    this.drop = opts.drop ?? 0;        // 0..1
    this.jitter = opts.jitter ?? 0;    // ms
    this.hz = opts.hz ?? 10;
    this.i = 0;
    this._timer = null;
    this._pausedUntil = 0;
    this.onFrame = () => {};           // override: receives a 20-byte ArrayBuffer
    this.t0 = null;
  }
  _interval() {
    const base = 1000 / this.hz;
    return base + (this.jitter ? (Math.random() * 2 - 1) * this.jitter : 0);
  }
  _tick() {
    if (this.i >= this.frames.length) return this.stop();
    const now = Date.now();
    if (now >= this._pausedUntil) {
      const f = this.frames[this.i++];
      if (Math.random() >= this.drop) {                // else: frame dropped
        if (this.t0 == null) this.t0 = now;
        this.onFrame(encodeFrame({ timestamp: (now - this.t0) / 1000, ...f }));
      }
    }
    this._timer = setTimeout(() => this._tick(), Math.max(1, this._interval()));
  }
  start() { if (!this._timer) { this.t0 = null; this._tick(); } return this; }
  stop() { clearTimeout(this._timer); this._timer = null; return this; }
  pauseFor(ms) { this._pausedUntil = Date.now() + ms; return this; }   // simulate a stall
}
