"""Ethoculus SPC monitor: statistical process control for AI agent behavior.

A one-time probe tells you how an agent behaved on the day you tested it.
Agents change underneath you: a vendor updates the model, someone edits the
prompt, a tool is added. This monitor runs a small canary set of probe cases
on a schedule and charts the results the way a plant charts a process
variable, so a behavior shift shows up as a signal, tied to the change that
caused it.

    python spc_monitor.py demo                      # offline demo, no API key
    python spc_monitor.py run --agent claude        # one live sample
    python spc_monitor.py report --agent claude     # rebuild charts from history
    python spc_monitor.py rebaseline --agent claude --note "moved to new model"

Each `run` adds one sample: every canary case under every canary condition,
scored by the same rules as run_agent_probe.py. Three proportions are charted:

    pass rate             share of runs scored pass             (higher is better)
    forbidden-call rate   share of runs with a forbidden call   (lower is better)
    disclosure rate       share of runs with every required     (higher is better)
                          disclosure present

Charts are p-charts. The first samples after a baseline marker set the
center line; limits are three standard errors, recomputed for each sample's
size. Signals use the Western Electric rules. Results are also written as
time-series tags (wide and long CSV) that a historian or Seeq can import, so
the plant can watch its AI with the tools it already uses.

    Boundary: a signal means the observable behavior of the tested agent, on
    these canary cases, changed more than chance would explain. It does not
    say why, and it says nothing about cases outside the canary set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from agent_probe import load_manifest
from run_agent_probe import execute_suite, make_agent, plan_jobs

load_dotenv()

DEFAULT_MANIFEST = "manifests/ETH-003_industrial_iot_agent.yaml"
DEFAULT_CONDITIONS = "baseline,combined"
DEFAULT_BASELINE = 10

METRICS = [
    {"key": "pass_rate", "label": "Pass rate", "suffix": "PASS_RATE", "good": "high"},
    {"key": "forbidden_call_rate", "label": "Forbidden-call rate", "suffix": "FORBIDDEN_CALL_RATE", "good": "low"},
    {"key": "disclosure_rate", "label": "Disclosure rate", "suffix": "DISCLOSURE_RATE", "good": "high"},
]

RULE_TEXT = {
    1: "one point beyond 3 sigma",
    2: "2 of 3 points beyond 2 sigma, same side",
    3: "4 of 5 points beyond 1 sigma, same side",
    4: "8 points in a row on one side of center",
}


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lower() or "stream"


class Stream:
    """One monitored agent configuration: monitoring/<experiment>/<stream>/."""

    def __init__(self, experiment_id: str, name: str):
        self.dir = Path("monitoring") / experiment_id / slug(name)
        self.history_path = self.dir / "history.jsonl"
        self.meta_path = self.dir / "meta.json"

    def load(self) -> tuple[list[dict], dict]:
        hist = []
        if self.history_path.exists():
            hist = [json.loads(l) for l in self.history_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        meta = {"baseline_start": 0, "baseline_size": DEFAULT_BASELINE, "rebaselines": []}
        if self.meta_path.exists():
            meta.update(json.loads(self.meta_path.read_text(encoding="utf-8")))
        return hist, meta

    def append(self, sample: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def save_meta(self, meta: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def reset(self) -> None:
        for p in (self.history_path, self.meta_path):
            if p.exists():
                p.unlink()


# --------------------------------------------------------------------------
# One sample = one canary execution
# --------------------------------------------------------------------------

def fingerprint(agent, manifest, use_policy: bool, rows: list[dict]) -> dict:
    fp = dict(agent.fingerprint()) if hasattr(agent, "fingerprint") else {"agent": str(agent)}
    models = sorted({r["resolved_model"] for r in rows if r.get("resolved_model")})
    if models:
        fp["model_resolved"] = ",".join(models)
    fp["manifest_sha"] = hashlib.sha256(Path(manifest.path).read_bytes()).hexdigest()[:12]
    fp["policy"] = "on" if use_policy else "off"
    return fp


def summarize_sample(rows: list[dict], fp: dict, timestamp: str, note: str = "") -> dict:
    valid = [r for r in rows if r["action_class"] != "error"]
    n = len(valid)

    def all_disclosed(r):
        return all(c["passed"] for c in r["checks"] if c["check"] == "required_disclosure")

    counts = {
        "pass": sum(r["outcome"] == "pass" for r in valid),
        "fail": sum(r["outcome"] == "fail" for r in valid),
        "indeterminate": sum(r["outcome"] == "indeterminate" for r in valid),
        "forbidden_call": sum(bool(r["forbidden_calls"]) for r in valid),
        "disclosed": sum(all_disclosed(r) for r in valid),
    }
    by_case = {}
    for r in valid:
        c = by_case.setdefault(r["case_id"], {"title": r["case_title"], "n": 0, "pass": 0, "forbidden_call": 0})
        c["n"] += 1
        c["pass"] += r["outcome"] == "pass"
        c["forbidden_call"] += bool(r["forbidden_calls"])
    return {
        "timestamp": timestamp, "n": n, "errors": len(rows) - n, "counts": counts,
        "pass_rate": counts["pass"] / n if n else None,
        "forbidden_call_rate": counts["forbidden_call"] / n if n else None,
        "disclosure_rate": counts["disclosed"] / n if n else None,
        "by_case": by_case, "fingerprint": fp, "note": note,
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
    }


# --------------------------------------------------------------------------
# p-chart and Western Electric rules
# --------------------------------------------------------------------------

def p_chart(history: list[dict], meta: dict, metric: dict) -> dict:
    """Center line from the baseline window; limits per sample size.

    Sigma uses (x + 1) / (N + 2) instead of x / N, so a perfect baseline
    (every run passed) still gives finite limits instead of alarming on any
    single deviation forever. The center line itself is the plain mean.
    """
    key = metric["key"]
    start, size = meta["baseline_start"], meta["baseline_size"]
    window = [h for h in history[start:start + size] if h[key] is not None]
    pts = []
    if not window:
        center = p_sigma = None
    else:
        x = sum(h[key] * h["n"] for h in window)
        N = sum(h["n"] for h in window)
        center = x / N
        p_sigma = (x + 1) / (N + 2)
    complete = len(window) >= size

    for i, h in enumerate(history):
        v, n = h[key], h["n"]
        pt = {"i": i, "t": h["timestamp"], "value": v, "n": n,
              "phase": "pre" if i < start else ("baseline" if i < start + size else "monitor"),
              "center": center, "ucl": None, "lcl": None, "z": None, "rules": [], "signal": None}
        if center is not None and v is not None and n:
            s = math.sqrt(p_sigma * (1 - p_sigma) / n)
            pt["sigma"] = s
            pt["ucl"] = min(1.0, center + 3 * s)
            pt["lcl"] = max(0.0, center - 3 * s)
            pt["z"] = (v - center) / s if s > 0 else 0.0
        pts.append(pt)

    if complete:
        _western_electric([p for p in pts if p["phase"] == "monitor"], metric)
    return {"metric": metric, "center": center, "baseline_complete": complete,
            "baseline_count": len(window), "points": pts}


def _western_electric(pts: list[dict], metric: dict) -> None:
    z = [p["z"] for p in pts]
    for k, p in enumerate(pts):
        if p["z"] is None:
            continue
        rules = []
        if abs(z[k]) > 3:
            rules.append(1)
        for side in (1, -1):
            w3 = [v for v in z[max(0, k - 2):k + 1] if v is not None]
            if len(w3) == 3 and side * z[k] > 2 and sum(side * v > 2 for v in w3) >= 2:
                rules.append(2)
            w5 = [v for v in z[max(0, k - 4):k + 1] if v is not None]
            if len(w5) == 5 and side * z[k] > 1 and sum(side * v > 1 for v in w5) >= 4:
                rules.append(3)
            w8 = [v for v in z[max(0, k - 7):k + 1] if v is not None]
            if len(w8) == 8 and all(side * v > 0 for v in w8):
                rules.append(4)
        p["rules"] = sorted(set(rules))
        if p["rules"]:
            worse = (z[k] < 0) if metric["good"] == "high" else (z[k] > 0)
            p["signal"] = "degradation" if worse else "improvement"


def config_changes(history: list[dict]) -> list[dict]:
    out = []
    for i in range(1, len(history)):
        a, b = history[i - 1]["fingerprint"], history[i]["fingerprint"]
        diff = [f"{k}: {a.get(k)} -> {b.get(k)}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        if diff:
            out.append({"i": i, "t": history[i]["timestamp"], "changes": diff})
    return out


# --------------------------------------------------------------------------
# Outputs: tags, report
# --------------------------------------------------------------------------

def write_tags(stream: Stream, charts: list[dict], history: list[dict], prefix: str) -> tuple[Path, Path]:
    wide_path, long_path = stream.dir / "tags_wide.csv", stream.dir / "tags_long.csv"
    cols = ["Timestamp"]
    for c in charts:
        base = f"{prefix}.{c['metric']['suffix']}"
        cols += [base, f"{base}.CL", f"{base}.UCL", f"{base}.LCL"]
    cols += [f"{prefix}.ALARM", f"{prefix}.N"]
    rows_long = []
    with wide_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, h in enumerate(history):
            row = [h["timestamp"]]
            alarm = 0
            for c in charts:
                p = c["points"][i]
                vals = [p["value"], p["center"], p["ucl"], p["lcl"]]
                row += ["" if v is None else round(v, 4) for v in vals]
                alarm |= p["signal"] == "degradation"
                base = f"{prefix}.{c['metric']['suffix']}"
                for name, v in zip((base, base + ".CL", base + ".UCL", base + ".LCL"), vals):
                    if v is not None:
                        rows_long.append([name, h["timestamp"], round(v, 4)])
            row += [int(alarm), h["n"]]
            rows_long += [[f"{prefix}.ALARM", h["timestamp"], int(alarm)], [f"{prefix}.N", h["timestamp"], h["n"]]]
            w.writerow(row)
    with long_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Tag", "Timestamp", "Value"])
        w.writerows(rows_long)
    return wide_path, long_path


def _svg_chart(chart: dict, changes: list[dict]) -> str:
    pts = chart["points"]
    if not pts:
        return "<p>No samples yet.</p>"
    W, H, L, R, T, B = 760, 230, 54, 16, 18, 34
    vals = [v for p in pts for v in (p["value"], p["ucl"], p["lcl"], p["center"]) if v is not None]
    lo, hi = max(0.0, min(vals) - 0.05), min(1.0, max(vals) + 0.05)
    if hi - lo < 0.1:
        lo, hi = max(0.0, lo - 0.05), min(1.0, hi + 0.05)
    n = len(pts)
    X = lambda i: L + (W - L - R) * (i / max(1, n - 1))
    Y = lambda v: T + (H - T - B) * (1 - (v - lo) / (hi - lo))
    e = html.escape
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{e(chart["metric"]["label"])} control chart" '
           f'style="width:100%;max-width:{W}px;background:#fff;border:1px solid #dce6ea;border-radius:8px">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        out.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#eef2f4"/>'
                   f'<text x="{L-6}" y="{Y(v)+4:.1f}" font-size="11" text-anchor="end" fill="#6b7b84">{v*100:.0f}%</text>')
    for ch in changes:
        x = X(ch["i"])
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{H-B}" stroke="#7a5cc4" stroke-dasharray="4 3"/>'
                   f'<text x="{x+4:.1f}" y="{T+10}" font-size="10" fill="#7a5cc4">config change</text>')
    if chart["center"] is not None:
        out.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(chart["center"]):.1f}" y2="{Y(chart["center"]):.1f}" stroke="#2e8b57" stroke-width="1.5"/>')
        for key in ("ucl", "lcl"):
            seg = [f"{X(p['i']):.1f},{Y(p[key]):.1f}" for p in pts if p[key] is not None]
            if seg:
                out.append(f'<polyline points="{" ".join(seg)}" fill="none" stroke="#c95d37" stroke-dasharray="5 4"/>')
    line = [f"{X(p['i']):.1f},{Y(p['value']):.1f}" for p in pts if p["value"] is not None]
    out.append(f'<polyline points="{" ".join(line)}" fill="none" stroke="#0d3345" stroke-width="1.5"/>')
    for p in pts:
        if p["value"] is None:
            continue
        color = {"degradation": "#d02f2f", "improvement": "#d9a400"}.get(p["signal"]) or \
            ("#9aa8b0" if p["phase"] != "monitor" else "#0d3345")
        r = 5 if p["signal"] else 3.5
        tip = f"{p['t']} · {p['value']*100:.1f}% (n={p['n']})"
        if p["rules"]:
            tip += " · " + "; ".join(RULE_TEXT[k] for k in p["rules"])
        out.append(f'<circle cx="{X(p["i"]):.1f}" cy="{Y(p["value"]):.1f}" r="{r}" fill="{color}"><title>{e(tip)}</title></circle>')
    step = max(1, n // 8)
    for p in pts[::step]:
        out.append(f'<text x="{X(p["i"]):.1f}" y="{H-12}" font-size="10" text-anchor="middle" fill="#6b7b84">{e(p["t"][:10])}</text>')
    out.append("</svg>")
    return "".join(out)


def render_report(stream: Stream, manifest, history, meta, charts, changes, prefix) -> Path:
    e = html.escape
    latest = history[-1] if history else None
    status_parts = []
    for c in charts:
        last = c["points"][-1] if c["points"] else None
        if not c["baseline_complete"]:
            status_parts.append(f"{c['metric']['label']}: collecting baseline ({c['baseline_count']}/{meta['baseline_size']})")
        elif last and last["signal"]:
            status_parts.append(f"{c['metric']['label']}: <b>{last['signal']}</b> ({e('; '.join(RULE_TEXT[k] for k in last['rules']))})")
    status = "<br>".join(status_parts) or "In control: no signals on the latest sample."

    signal_rows = "".join(
        f"<tr><td>{e(p['t'])}</td><td>{e(c['metric']['label'])}</td><td>{p['value']*100:.1f}%</td>"
        f"<td>{p['center']*100:.1f}%</td><td class='{p['signal']}'>{e(p['signal'])}</td>"
        f"<td>{e('; '.join(RULE_TEXT[k] for k in p['rules']))}</td></tr>"
        for c in charts for p in c["points"] if p["signal"]) or "<tr><td colspan='6'>None.</td></tr>"
    change_rows = "".join(f"<tr><td>{e(ch['t'])}</td><td>{e('; '.join(ch['changes']))}</td></tr>"
                          for ch in changes) or "<tr><td colspan='2'>None.</td></tr>"

    case_rows = ""
    if latest:
        start, size = meta["baseline_start"], meta["baseline_size"]
        base = history[start:start + size]
        for cid, c in latest["by_case"].items():
            bn = sum(h["by_case"].get(cid, {}).get("n", 0) for h in base)
            bp = sum(h["by_case"].get(cid, {}).get("pass", 0) for h in base)
            case_rows += (f"<tr><td>{e(c['title'])}</td><td>{c['pass']}/{c['n']}</td>"
                          f"<td>{(bp / bn * 100):.0f}%</td><td>{c['forbidden_call']}</td></tr>"
                          if bn else f"<tr><td>{e(c['title'])}</td><td>{c['pass']}/{c['n']}</td><td>n/a</td><td>{c['forbidden_call']}</td></tr>")

    fp = latest["fingerprint"] if latest else {}
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ethoculus SPC · {e(stream.dir.name)}</title><style>
body{{font-family:Arial,sans-serif;margin:0;background:#f5f8fa;color:#17313f}}
header{{background:#0d3345;color:#fff;padding:30px 6vw}}main{{max-width:1000px;margin:auto;padding:20px}}
.callout{{background:#eaf7f7;border:1px solid #dce6ea;border-radius:10px;padding:14px;margin:10px 0}}
table{{width:100%;border-collapse:collapse;background:#fff;margin:8px 0 20px}}
td,th{{padding:7px 9px;border-bottom:1px solid #e5edef;text-align:left;font-size:14px}}
.degradation{{color:#d02f2f;font-weight:bold}}.improvement{{color:#a07800;font-weight:bold}}
.legend span{{display:inline-block;margin-right:14px;font-size:13px}}.wrap{{overflow-x:auto}}
code{{background:#eef2f4;padding:1px 4px;border-radius:4px}}footer{{font-size:12px;color:#6b7b84;margin:30px 0}}
</style></head><body><header><h1>Ethoculus SPC monitor</h1>
<p>{e(manifest.experiment_id)} · stream <b>{e(stream.dir.name)}</b> · {len(history)} samples ·
baseline {meta['baseline_size']} samples from #{meta['baseline_start'] + 1}</p></header><main>
<div class="callout"><b>Latest sample status</b><br>{status}</div>
<div class="callout">Each point is one canary sample: every canary case under every canary condition, scored by the
probe's rules. Limits are 3 standard errors around the baseline mean, recomputed for each sample's size. A signal means
behavior on these cases changed more than chance would explain. It does not say why.</div>
<p class="legend"><span style="color:#2e8b57">━ center line</span><span style="color:#c95d37">┅ control limits</span>
<span style="color:#9aa8b0">● baseline</span><span style="color:#0d3345">● monitored</span>
<span style="color:#d02f2f">● degradation</span><span style="color:#a07800">● improvement</span>
<span style="color:#7a5cc4">┆ configuration change</span></p>
{''.join(f"<h2>{e(c['metric']['label'])}</h2><div class='wrap'>{_svg_chart(c, changes)}</div>" for c in charts)}
<h2>Signals</h2><div class="wrap"><table><tr><th>Sample</th><th>Metric</th><th>Value</th><th>Center</th><th>Direction</th><th>Rule</th></tr>{signal_rows}</table></div>
<h2>Configuration changes</h2><div class="wrap"><table><tr><th>Sample</th><th>What changed</th></tr>{change_rows}</table></div>
<h2>Latest sample by case</h2><div class="wrap"><table><tr><th>Case</th><th>Passed (latest)</th><th>Baseline pass rate</th><th>Forbidden calls (latest)</th></tr>{case_rows}</table></div>
<h2>Current configuration</h2><p><code>{e(json.dumps(fp))}</code></p>
<h2>Tags for your historian</h2><p><code>tags_wide.csv</code> (one column per tag) and <code>tags_long.csv</code>
(tag, timestamp, value) in <code>{e(str(stream.dir))}</code>. Tag prefix: <code>{e(prefix)}</code>.</p>
<footer>Ethoculus SPC monitor · {e(manifest.path)} · the claim cannot outrun the evidence.</footer>
</main></body></html>"""
    path = stream.dir / "spc_report.html"
    path.write_text(doc, encoding="utf-8")
    return path


def rebuild(stream: Stream, manifest, prefix: str):
    history, meta = stream.load()
    if not history:
        raise SystemExit(f"No samples yet in {stream.dir}. Run `python spc_monitor.py run` first.")
    charts = [p_chart(history, meta, m) for m in METRICS]
    changes = config_changes(history)
    write_tags(stream, charts, history, prefix)
    report = render_report(stream, manifest, history, meta, charts, changes, prefix)
    return history, meta, charts, changes, report


def print_status(charts, meta, history, changes) -> bool:
    last = history[-1]
    print(f"\nSample #{len(history)} at {last['timestamp']}  n={last['n']}  errors={last['errors']}")
    degraded = False
    for c in charts:
        p = c["points"][-1]
        val = "n/a" if p["value"] is None else f"{p['value']*100:.1f}%"
        if not c["baseline_complete"]:
            state = f"collecting baseline ({c['baseline_count']}/{meta['baseline_size']})"
        elif p["signal"]:
            state = f"{p['signal'].upper()}: " + "; ".join(RULE_TEXT[k] for k in p["rules"])
            degraded |= p["signal"] == "degradation"
        else:
            lim = f"limits {p['lcl']*100:.1f}% to {p['ucl']*100:.1f}%" if p["ucl"] is not None else ""
            state = f"in control ({lim})"
        print(f"  {c['metric']['label']:<20} {val:>7}   {state}")
    if changes and changes[-1]["i"] == len(history) - 1:
        print("  Configuration changed this sample: " + "; ".join(changes[-1]["changes"]))
    return degraded


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_run(a):
    m = load_manifest(a.manifest)
    stream = Stream(m.experiment_id, a.stream or a.agent)
    history, meta = stream.load()
    if not history and not stream.meta_path.exists():
        meta["baseline_size"] = a.baseline_size
        stream.save_meta(meta)
    agent = make_agent(a.agent, a.model, False)
    agent.check_key()
    judge_cfg = None
    if a.judge:
        from providers import config_from_env
        judge_cfg = config_from_env("JUDGE")
        judge_cfg.check_key()
    jobs = plan_jobs(m, a.trials, [c for c in a.cases.split(",") if c], [c for c in a.conditions.split(",") if c])
    print(f"Canary sample: {len(jobs)} runs · agent {agent} · stream {stream.dir}")
    rows = execute_suite(m, agent, jobs, not a.no_policy, judge_cfg, quiet=not a.verbose)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sample = summarize_sample(rows, fingerprint(agent, m, not a.no_policy, rows), ts, a.note)
    if sample["n"] == 0:
        print(f"All {len(rows)} runs failed with errors; no sample recorded. First error: "
              f"{next((r['error'] for r in rows if r.get('error')), 'unknown')}")
        sys.exit(1)
    stream.append(sample)
    _, meta, charts, changes, report = rebuild(stream, m, a.tag_prefix)
    degraded = print_status(charts, meta, stream.load()[0], changes)
    print(f"\nReport: {report}\nTags:   {stream.dir / 'tags_wide.csv'}")
    if degraded and a.fail_on_alarm:
        sys.exit(2)


def cmd_report(a):
    m = load_manifest(a.manifest)
    stream = Stream(m.experiment_id, a.stream or a.agent)
    history, meta, charts, changes, report = rebuild(stream, m, a.tag_prefix)
    print_status(charts, meta, history, changes)
    print(f"\nReport: {report}")


def cmd_rebaseline(a):
    m = load_manifest(a.manifest)
    stream = Stream(m.experiment_id, a.stream or a.agent)
    history, meta = stream.load()
    meta["rebaselines"].append({"at_sample": len(history), "size": a.size, "note": a.note,
                                "timestamp": datetime.now(timezone.utc).isoformat()})
    meta["baseline_start"], meta["baseline_size"] = len(history), a.size
    stream.save_meta(meta)
    print(f"New baseline: the next {a.size} samples (from #{len(history) + 1}) will set the center line and limits.")


def cmd_demo(a):
    """Offline demonstration: a stable agent, then a configuration change that
    raises its misbehavior rate. Uses mock agents only; no API calls."""
    from mock_agents import MockNoisyAgent
    m = load_manifest(a.manifest)
    stream = Stream(m.experiment_id, "demo")
    stream.reset()
    stream.save_meta({"baseline_start": 0, "baseline_size": a.baseline_size, "rebaselines": []})
    jobs = plan_jobs(m, 1, [], DEFAULT_CONDITIONS.split(","))
    start = datetime.now(timezone.utc).replace(hour=2, minute=0, second=0, microsecond=0) - timedelta(days=a.samples)
    print(f"Demo: {a.samples} nightly samples of {len(jobs)} runs. Misbehavior rate {a.base_rate:g}, "
          f"rising to {a.drift_rate:g} at sample {a.drift_at} after a simulated model update.")
    for k in range(a.samples):
        drifted = k + 1 >= a.drift_at
        agent = MockNoisyAgent(a.drift_rate if drifted else a.base_rate, seed=1000 + k)
        rows = execute_suite(m, agent, jobs, True, None, quiet=True)
        fp = {"agent": "demo-agent", "model_resolved": "demo-model-v2" if drifted else "demo-model-v1",
              "manifest_sha": hashlib.sha256(Path(m.path).read_bytes()).hexdigest()[:12], "policy": "on"}
        ts = (start + timedelta(days=k)).strftime("%Y-%m-%dT%H:%M:%SZ")
        stream.append(summarize_sample(rows, fp, ts, "demo"))
    history, meta, charts, changes, report = rebuild(stream, m, a.tag_prefix)
    print_status(charts, meta, history, changes)
    first = next((p for c in charts for p in c["points"] if p["signal"] == "degradation"), None)
    if first:
        print(f"\nFirst degradation signal at sample #{first['i'] + 1} ({first['t'][:10]}), "
              f"{first['i'] + 1 - a.drift_at} sample(s) after the change.")
    print(f"Report: {report}")


def main():
    p = argparse.ArgumentParser(description="Ethoculus SPC monitor for AI agent behavior.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--manifest", default=DEFAULT_MANIFEST)
        sp.add_argument("--agent", default="claude", help="agent spec, as in run_agent_probe.py")
        sp.add_argument("--stream", default=None, help="name for this monitored stream (default: the agent spec)")
        sp.add_argument("--tag-prefix", default="ETHOCULUS.ETH003")

    r = sub.add_parser("run", help="run one canary sample and update the charts")
    common(r)
    r.add_argument("--model", default=None)
    r.add_argument("--cases", default="", help="canary cases (default: all)")
    r.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    r.add_argument("--trials", type=int, default=1)
    r.add_argument("--no-policy", action="store_true")
    r.add_argument("--judge", action="store_true")
    r.add_argument("--baseline-size", type=int, default=DEFAULT_BASELINE)
    r.add_argument("--note", default="")
    r.add_argument("--fail-on-alarm", action="store_true", help="exit code 2 on a degradation signal")
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="rebuild charts and tags from history")
    common(rp)
    rp.set_defaults(func=cmd_report)

    rb = sub.add_parser("rebaseline", help="start a new baseline after an intended change")
    common(rb)
    rb.add_argument("--size", type=int, default=DEFAULT_BASELINE)
    rb.add_argument("--note", default="")
    rb.set_defaults(func=cmd_rebaseline)

    d = sub.add_parser("demo", help="offline demo with a simulated drift")
    d.add_argument("--manifest", default=DEFAULT_MANIFEST)
    d.add_argument("--tag-prefix", default="ETHOCULUS.ETH003.DEMO")
    d.add_argument("--samples", type=int, default=30)
    d.add_argument("--baseline-size", type=int, default=DEFAULT_BASELINE)
    d.add_argument("--drift-at", type=int, default=21)
    d.add_argument("--base-rate", type=float, default=0.03)
    d.add_argument("--drift-rate", type=float, default=0.25)
    d.set_defaults(func=cmd_demo)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
