/**
 * A6 — agent_trace recorder. n8n Code node, "Run Once for All Items".
 * Sits after the Supervisor Agent (which must have "Return Intermediate Steps" ON).
 * Turns the Supervisor's tool calls into an ordered conversation log, pulls out
 * each specialist's structured result, and counts calls per agent.
 *
 * n8n gives the agent result as { output, intermediateSteps }. Each step looks like
 *   { action: { tool, toolInput, log }, observation }
 * where observation is the tool's return (object, or a JSON string). We parse both.
 */

const root = $input.first().json || {};
const steps = Array.isArray(root.intermediateSteps) ? root.intermediateSteps
            : Array.isArray(root.output?.intermediateSteps) ? root.output.intermediateSteps
            : [];

const asObj = (v) => {
  if (v == null) return null;
  if (typeof v === "object") return v.json ?? v;     // n8n sometimes wraps as {json:...}
  if (typeof v === "string") { try { return JSON.parse(v); } catch { return { text: v }; } }
  return { value: v };
};
const oneLine = (v) => {
  const o = asObj(v);
  if (o == null) return "";
  if (o.text) return String(o.text).slice(0, 160);
  return JSON.stringify(o).slice(0, 160);
};
const toolOf = (a = {}) => a.tool || a.toolName || (a.action && a.action.tool) || "unknown";
const inputOf = (a = {}) => a.toolInput ?? a.input ?? (a.action && a.action.toolInput) ?? {};

const agent_trace = [];
const calls_per_agent = {};
const results = {};               // latest parsed result per tool

steps.forEach((s, i) => {
  const action = s.action || s;
  const tool = toolOf(action);
  const obs = s.observation !== undefined ? s.observation : s.result;
  const parsed = asObj(obs);

  calls_per_agent[tool] = (calls_per_agent[tool] || 0) + 1;
  if (parsed && typeof parsed === "object") results[tool] = parsed;

  agent_trace.push({
    step: i + 1,
    from: "supervisor",
    to: tool,
    message: typeof inputOf(action) === "string" ? inputOf(action) : JSON.stringify(inputOf(action)).slice(0, 160),
    reply_summary: oneLine(obs),
  });
});

// Map tool names to roles, however they were labelled.
const pick = (...keys) => { for (const k of keys) if (results[k]) return results[k]; return null; };

const out = {
  supervisor_summary: typeof root.output === "string" ? root.output : (root.output?.text ?? ""),
  agent_trace,
  calls_per_agent,
  assessment: pick("call_assessment", "Assessment", "A3 - Assessment Agent"),
  progress:   pick("call_progress", "Progress", "A4 - Progress Agent"),
  coaching:   pick("call_coaching", "Coaching", "Coaching Stub"),
  review:     pick("call_safety", "Safety", "Safety Stub"),
  case_file:  pick("get_case_file"),
  // handy flags for the A7 conversation checks:
  called_fault_history_anywhere: agent_trace.some(t => /fault_history/i.test(t.to)),
  n_agent_calls: agent_trace.length,
};

return [{ json: out }];
