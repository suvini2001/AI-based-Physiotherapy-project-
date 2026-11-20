/**
 * metrics.js — n8n Code node, run mode "Run Once for All Items".
 * A1 (analysis path). Computes every number the agents and the rule-flags node read.
 *
 * Input:  one session.completed event (see agents/contract/session_event.v1.schema.json)
 * Output: { ev, metrics } — get_case_file adds the rule flags and prescription on top of this.
 *
 * It reads the event whether it arrives wrapped as { body: {...} } (behind the webhook)
 * or bare (when you pin a fixture and run it from a Manual Trigger), so the same node
 * works in the live flow and in standalone tests.
 */

// ── thresholds: placeholders until a physio agrees them; they live ONLY here ──
const CONFIG = {
    fastRepMaxSeconds: 1.0,                 // a rep shorter than this counts as "fast"
    baselineSessions: 3,                   // how many past sessions define the baseline
    slopeMinSessions: 3,                   // need at least this many points for a trend slope
    quality: { good: 0.75, fair: 0.50 },    // mean_quality >= good -> "good", >= fair -> "fair", else "poor"
};

// ── read the event, wrapped or bare ──
const input = $input.first().json;    //n8n node එකට ලැබුණු සම්පූර්ණ JSON
const ev = input && input.body ? input.body : input;   //Session completed event එක
const s = ev.session || {};         //Current session details
const rx = ev.prescription || {};   //Physiotherapistගේ prescription targets
const reps = Array.isArray(s.reps) ? s.reps : [];   //	Current session එකේ repetitions
const hist = Array.isArray(ev.history) ? ev.history : [];   // newest first (history[0] = most recent past session)

// ── helpers ──
const round0 = (x) => (x === null ? null : Math.round(x));   //Decimal places නැතුව round කරනවා.
const round1 = (x) => (x === null ? null : +x.toFixed(1));   //Decimal place එකකට round කරනවා.
const round2 = (x) => (x === null ? null : +x.toFixed(2));   //Decimal places දෙකකට round කරනවා.
const pct = (num, den) => (den ? (100 * num) / den : null);   //  Percentage calculate කරනවා. null, never Infinity, when target is 0 or missing
const isNum = (v) => typeof v === "number" && !Number.isNaN(v);

// ── fault counts across this session's reps, keyed by stable fault code ──
const faultCounts = {};
for (const r of reps) {
    for (const f of (r.faults || [])) {
        if (f && f.code) faultCounts[f.code] = (faultCounts[f.code] || 0) + 1;
    }
}

// ── fast-rep share ──
const fastReps = reps.filter((r) => isNum(r.duration_s) && r.duration_s < CONFIG.fastRepMaxSeconds).length;
const fastRepShare = reps.length ? round2(fastReps / reps.length) : null;

// ── ROM slope over history + this session (least squares, oldest first) ──
// series goes oldest -> newest: past sessions (history reversed) then this session.
const series = [...hist].reverse()
    .map((h) => h.mean_rom)
    .concat([s.mean_rom])
    .filter(isNum);
let slope = null;
const n = series.length;
if (n >= CONFIG.slopeMinSessions) {
    const xMean = (n - 1) / 2;
    const yMean = series.reduce((a, b) => a + b, 0) / n;
    let num = 0, den = 0;
    series.forEach((y, x) => { num += (x - xMean) * (y - yMean); den += (x - xMean) ** 2; });
    slope = den ? round2(num / den) : null;
}

// ── baseline: mean ROM of the N sessions just before this one (history is newest first) ──
const baseWin = hist.slice(0, CONFIG.baselineSessions).map((h) => h.mean_rom).filter(isNum);
const baselineRom = baseWin.length ? round1(baseWin.reduce((a, b) => a + b, 0) / baseWin.length) : null;

// ── quality band from mean_quality ──
let qualityBand = null;
if (isNum(s.mean_quality)) {
    qualityBand = s.mean_quality >= CONFIG.quality.good ? "good"
        : s.mean_quality >= CONFIG.quality.fair ? "fair"
            : "poor";
}

// ── assemble ──
const metrics = {
    reps_vs_target_pct: round0(pct(s.total_reps, rx.target_reps)),
    rom_vs_target_pct: round0(pct(s.mean_rom, rx.target_rom)),
    fault_counts: faultCounts,
    fast_rep_share: fastRepShare,
    rom_slope_deg_per_session: slope,          // null when fewer than slopeMinSessions points
    baseline_rom: baselineRom,     // null when there is no history
    quality_band: qualityBand,     // null when mean_quality is missing
    incomplete_attempts: isNum(s.incomplete_attempts) ? s.incomplete_attempts : 0,
    sessions_in_history: hist.length,
};

return [{ json: { ev, metrics } }];