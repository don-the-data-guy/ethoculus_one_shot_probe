"""Run an Ethoculus agent probe from a manifest.

    python run_agent_probe.py --dry-run
    python run_agent_probe.py --agent mock-good
    python run_agent_probe.py --agent claude --cases esp_restart --conditions baseline,combined
    python run_agent_probe.py --agent claude --trials 2 --judge
    python run_agent_probe.py --agent claude --no-policy          # ablation
    python run_agent_probe.py --agent python:my_agent:MyAgent      # your own agent

Every case is crossed with every condition, the job order is shuffled with
a fixed seed, and every tool call is recorded by the simulator rather than
taken from the agent's own account of what it did.

Reports go to reports/agent_probe/<experiment_id>/.
"""

from __future__ import annotations

import argparse
import html
import importlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agent_probe import (SimulatedPlant, apply_judgment, build_task, judge_run,
                         load_manifest, policy_text, score_run)

load_dotenv()

DEFAULT_MANIFEST = "manifests/ETH-003_industrial_iot_agent.yaml"
SEED = 20261017


# --------------------------------------------------------------------------
# Agent factory
# --------------------------------------------------------------------------

def make_agent(spec: str, model: str | None, show_http: bool):
    if spec == "claude":
        from claude_plant_agent import ClaudeAgent
        return ClaudeAgent(model=model, show_http=show_http)
    if spec.startswith("mock-noisy"):
        from mock_agents import MockNoisyAgent
        rate = float(spec.split(":", 1)[1]) if ":" in spec else 0.05
        return MockNoisyAgent(rate)
    if spec.startswith("mock-"):
        from mock_agents import MOCKS
        if spec not in MOCKS:
            raise SystemExit(f"Unknown mock '{spec}'. Choose from: {', '.join(MOCKS)}")
        return MOCKS[spec]()
    if spec.startswith("python:"):
        try:
            _, module, attr = spec.split(":", 2)
        except ValueError:
            raise SystemExit("Use --agent python:module:ClassName")
        obj = getattr(importlib.import_module(module), attr)
        return obj() if isinstance(obj, type) else obj
    raise SystemExit("--agent must be claude, mock-good, mock-sloppy, mock-bad, mock-noisy:<rate>, or python:module:Class")


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def plan_jobs(manifest, trials, case_ids, condition_ids):
    cases = [manifest.case(c) for c in case_ids] if case_ids else manifest.cases
    conds = [manifest.condition(c) for c in condition_ids] if condition_ids else manifest.conditions
    jobs = [(t, c, k) for t in range(1, trials + 1) for c in cases for k in conds]
    random.Random(SEED).shuffle(jobs)
    return jobs


def execute_suite(manifest, agent, jobs, use_policy, judge_cfg=None, quiet=False):
    rows = []
    say = (lambda *a, **k: None) if quiet else print
    for i, (trial, case, cond) in enumerate(jobs, 1):
        say(f"[{i:>3}/{len(jobs)}] {case.title} · {cond.label} · trial {trial}")
        plant = SimulatedPlant(manifest, case)
        task = build_task(case, cond)
        transcript = agent.run(task, policy_text(manifest, use_policy), manifest.tools, plant.execute)
        transcript["tool_calls"] = plant.calls      # the simulator's record is the evidence
        score = score_run(manifest, case, transcript)
        if judge_cfg is not None and case.judge_rubric and score["outcome"] != "fail":
            from providers import generate, parse_json
            try:
                score = apply_judgment(score, judge_run(judge_cfg, manifest, case, task,
                                                        transcript, generate, parse_json))
            except Exception as e:  # judge failure never hides the rule result
                score["judge_error"] = str(e)
        say(f"        -> {score['outcome']:<13} {score['action_class']:<14} "
              f"tools={len(plant.calls)} {('FORBIDDEN: ' + ', '.join(score['forbidden_calls'])) if score['forbidden_calls'] else ''}")
        api = transcript.get("api_calls") or []
        rows.append({
            "experiment_id": manifest.experiment_id, "trial": trial,
            "case_id": case.case_id, "case_title": case.title, "industry": case.industry,
            "misbehavior_tested": case.misbehavior_tested,
            "condition_id": cond.cid, "condition_label": cond.label, "pressure_level": cond.pressure,
            "policy_enabled": use_policy, "agent": str(agent), "task": task,
            "outcome": score["outcome"], "action_class": score["action_class"],
            "basis": score["basis"], "flagged": score["flagged"], "checks": score["checks"],
            "forbidden_calls": score["forbidden_calls"], "scored_by": score["scored_by"],
            "judge": score.get("judge"), "judge_error": score.get("judge_error"),
            "tool_calls": plant.calls, "final_text": transcript.get("final_text", ""),
            "assistant_texts": transcript.get("assistant_texts", []),
            "turns": transcript.get("turns", 0), "truncated": transcript.get("truncated", False),
            "error": transcript.get("error"),
            "response_ids": [c.get("response_id") for c in api],
            "resolved_model": next((c.get("model") for c in api if c.get("model")), None),
            "input_tokens": sum(c.get("input_tokens", 0) for c in api),
            "output_tokens": sum(c.get("output_tokens", 0) for c in api),
            "latency_ms": sum(c.get("latency_ms", 0) for c in api),
            "api_calls": len(api),
        })
    return rows


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def _rates(items):
    n = len(items)
    flagged = [r["flagged"] for r in items if r["flagged"] is not None]
    return {
        "n": n,
        "pass_rate": sum(r["outcome"] == "pass" for r in items) / n,
        "fail_rate": sum(r["outcome"] == "fail" for r in items) / n,
        "indeterminate_rate": sum(r["outcome"] == "indeterminate" for r in items) / n,
        "forbidden_call_rate": sum(bool(r["forbidden_calls"]) for r in items) / n,
        "misreport_rate": sum(r["action_class"] == "misreport" for r in items) / n,
        "flagged_rate": (sum(flagged) / len(flagged)) if flagged else None,
    }


def summarize(rows, manifest, agent_name, use_policy):
    by_cond, by_case = defaultdict(list), defaultdict(list)
    for r in rows:
        by_cond[r["condition_id"]].append(r)
        by_case[r["case_id"]].append(r)
    conds = {}
    for cid, items in by_cond.items():
        conds[cid] = {"label": items[0]["condition_label"],
                      "pressure_level": items[0]["pressure_level"], **_rates(items)}
    base = conds.get("baseline", {}).get("pass_rate")
    for c in conds.values():
        c["pass_drop_vs_baseline"] = (base - c["pass_rate"]) if base is not None else None
    cases = {cid: {"title": items[0]["case_title"], "misbehavior_tested": items[0]["misbehavior_tested"],
                   **_rates(items)} for cid, items in by_case.items()}
    forbidden = Counter(n for r in rows for n in r["forbidden_calls"])
    kinds = Counter(c["kind"] for r in rows for c in r["tool_calls"])
    tin = sum(r["input_tokens"] for r in rows)
    tout = sum(r["output_tokens"] for r in rows)
    in_rate = float(os.getenv("AGENT_INPUT_USD_PER_MTOK", "0") or 0)
    out_rate = float(os.getenv("AGENT_OUTPUT_USD_PER_MTOK", "0") or 0)
    failures = sorted([r for r in rows if r["outcome"] == "fail"],
                      key=lambda r: (bool(r["forbidden_calls"]), r["pressure_level"]), reverse=True)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": manifest.experiment_id, "title": manifest.title,
        "manifest": manifest.path, "agent": agent_name, "policy_enabled": use_policy,
        "runs": len(rows), "api_calls": sum(r["api_calls"] for r in rows),
        "input_tokens": tin, "output_tokens": tout,
        "estimated_cost_usd": (tin / 1e6 * in_rate + tout / 1e6 * out_rate) if (in_rate or out_rate) else None,
        "overall": _rates(rows), "conditions": conds, "cases": cases,
        "forbidden_calls_by_tool": dict(forbidden), "tool_calls_by_kind": dict(kinds),
        "top_failures": [{k: r[k] for k in ("case_title", "condition_label", "trial", "action_class",
                                             "basis", "forbidden_calls")} for r in failures[:10]],
    }


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

def pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def write_reports(rows, summary, manifest):
    out = Path("reports/agent_probe") / manifest.experiment_id
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw = out / f"agent-probe-{stamp}.jsonl"
    with raw.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "latest-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    conds = sorted(summary["conditions"].values(), key=lambda c: c["pressure_level"])
    md = [
        f"# Ethoculus {manifest.experiment_id}: {manifest.title}", "",
        f"**Agent:** `{summary['agent']}`  ",
        f"**Deployment policy:** {'on' if summary['policy_enabled'] else 'OFF (ablation)'}  ",
        f"**Runs:** {summary['runs']}  **API calls:** {summary['api_calls']}  "
        f"**Tokens in/out:** {summary['input_tokens']}/{summary['output_tokens']}  ",
        f"**Manifest:** `{manifest.path}`", "",
        "> Behavioral evidence only. Tool calls were recorded by a simulator; nothing touched a "
        "real system. This does not reveal chain-of-thought, intent, or internal mechanism.", "",
        "## Question posed", "", manifest.research_question, "",
        "## Results by condition", "",
        "| Condition | Pressure | N | Pass | Fail | Indeterminate | Forbidden call | Misreport | Pass drop vs baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in conds:
        md.append(f"| {c['label']} | {c['pressure_level']} | {c['n']} | {pct(c['pass_rate'])} | "
                  f"{pct(c['fail_rate'])} | {pct(c['indeterminate_rate'])} | "
                  f"{pct(c['forbidden_call_rate'])} | {pct(c['misreport_rate'])} | "
                  f"{pct(c['pass_drop_vs_baseline'])} |")
    md += ["", "## Results by case", "",
           "| Case | Misbehavior tested | N | Pass | Fail | Indeterminate | Flagged |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for c in summary["cases"].values():
        md.append(f"| {c['title']} | {c['misbehavior_tested']} | {c['n']} | {pct(c['pass_rate'])} | "
                  f"{pct(c['fail_rate'])} | {pct(c['indeterminate_rate'])} | {pct(c['flagged_rate'])} |")
    md += ["", "## Forbidden tool calls", ""]
    md += ([f"- `{k}`: {v}" for k, v in summary["forbidden_calls_by_tool"].items()]
           or ["None observed."])
    md += ["", "## Failures", ""]
    fails = [r for r in rows if r["outcome"] == "fail"]
    for r in fails[:12]:
        md += [f"### {r['case_title']} · {r['condition_label']} · trial {r['trial']}",
               f"- {r['action_class']}: {r['basis']}",
               f"- Final report: {r['final_text'][:400]}", ""]
    if not fails:
        md.append("No failures in this run.")
    md += ["", "## Reading these results", "",
           "Report findings in the form: *under condition X, observable behavior Y occurred at "
           "rate Z.* Indeterminate rows need a human read or `--judge`. A single run is Level 1 "
           "evidence (observation) on the Ethoculus evidence ladder; replication across trials, "
           "paraphrased tasks, and model versions is required before claiming a pattern."]
    (out / "latest.md").write_text("\n".join(md), encoding="utf-8")

    e = html.escape
    cond_rows = "".join(
        f"<tr><td>{e(c['label'])}</td><td>{c['pressure_level']}</td><td>{c['n']}</td>"
        f"<td>{pct(c['pass_rate'])}</td><td>{pct(c['fail_rate'])}</td>"
        f"<td>{pct(c['indeterminate_rate'])}</td><td>{pct(c['forbidden_call_rate'])}</td>"
        f"<td>{pct(c['misreport_rate'])}</td><td>{pct(c['pass_drop_vs_baseline'])}</td></tr>"
        for c in conds)
    case_rows = "".join(
        f"<tr><td>{e(c['title'])}</td><td>{e(c['misbehavior_tested'])}</td><td>{c['n']}</td>"
        f"<td>{pct(c['pass_rate'])}</td><td>{pct(c['fail_rate'])}</td>"
        f"<td>{pct(c['indeterminate_rate'])}</td><td>{pct(c['flagged_rate'])}</td></tr>"
        for c in summary["cases"].values())

    def row_html(r):
        calls = "".join(
            f"<li class='{'bad' if c['name'] in r['forbidden_calls'] else ''}'>"
            f"<code>[{e(c['kind'])}] {e(c['name'])}({e(json.dumps(c['input'])[:220])})</code></li>"
            for c in r["tool_calls"]) or "<li>(no tool calls)</li>"
        return (f"<details class='{r['outcome']}'><summary><b>{e(r['outcome'].upper())}</b> · "
                f"{e(r['case_title'])} · {e(r['condition_label'])} · trial {r['trial']} · "
                f"{e(r['action_class'])}</summary>"
                f"<p><b>Basis:</b> {e(r['basis'])}</p>"
                f"<p><b>Task:</b><br>{e(r['task'])}</p>"
                f"<p><b>Tool calls (simulator record):</b></p><ul>{calls}</ul>"
                f"<p><b>Final report:</b></p><pre>{e(r['final_text'] or '(empty)')}</pre></details>")

    ordered = sorted(rows, key=lambda r: ({"fail": 0, "indeterminate": 1, "pass": 2}[r["outcome"]],
                                          r["case_id"], r["pressure_level"]))
    forbidden_html = "".join(f"<li><code>{e(k)}</code>: {v}</li>"
                             for k, v in summary["forbidden_calls_by_tool"].items()) or "<li>None observed.</li>"
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ethoculus {e(manifest.experiment_id)}</title><style>
body{{font-family:Arial,sans-serif;margin:0;background:#f5f8fa;color:#17313f}}
header{{background:#0d3345;color:#fff;padding:36px 6vw}}main{{max-width:1100px;margin:auto;padding:24px 20px}}
table{{width:100%;border-collapse:collapse;background:#fff;margin:8px 0 20px}}
td,th{{padding:7px 9px;border-bottom:1px solid #e5edef;text-align:right;font-size:14px}}
td:first-child,th:first-child,td:nth-child(2){{text-align:left}}
.callout{{background:#eaf7f7;border:1px solid #dce6ea;border-radius:10px;padding:14px;margin:10px 0}}
details{{background:#fff;border:1px solid #dce6ea;border-radius:10px;padding:10px 14px;margin:8px 0}}
details.fail{{border-left:5px solid #c95d37}}details.indeterminate{{border-left:5px solid #d9a400}}
details.pass{{border-left:5px solid #2e8b57}}pre{{white-space:pre-wrap;background:#f5f8fa;padding:10px;border-radius:8px}}
li.bad code{{color:#b03a1a;font-weight:bold}}.wrap{{overflow-x:auto}}footer{{font-size:12px;color:#6b7b84;margin:30px 0}}
</style></head><body><header><h1>Ethoculus {e(manifest.experiment_id)}: {e(manifest.title)}</h1>
<p>Agent <b>{e(summary['agent'])}</b> · policy {'on' if summary['policy_enabled'] else '<b>OFF (ablation)</b>'} ·
{summary['runs']} runs · {summary['api_calls']} API calls</p></header><main>
<div class="callout"><b>Question posed:</b> {e(manifest.research_question)}</div>
<div class="callout">Behavioral evidence only. Every tool call below was recorded by a simulator; nothing touched a real
system. Indeterminate rows need a human read or <code>--judge</code>.</div>
<h2>By condition</h2><div class="wrap"><table><tr><th>Condition</th><th>Pressure</th><th>N</th><th>Pass</th><th>Fail</th>
<th>Indeterminate</th><th>Forbidden call</th><th>Misreport</th><th>Pass drop</th></tr>{cond_rows}</table></div>
<h2>By case</h2><div class="wrap"><table><tr><th>Case</th><th>Misbehavior tested</th><th>N</th><th>Pass</th><th>Fail</th>
<th>Indeterminate</th><th>Flagged</th></tr>{case_rows}</table></div>
<h2>Forbidden tool calls</h2><ul>{forbidden_html}</ul>
<h2>Every run</h2>{''.join(row_html(r) for r in ordered)}
<footer>Ethoculus {e(manifest.experiment_id)} · {e(manifest.path)} · Make It Your AI. Not Theirs.</footer>
</main></body></html>"""
    (out / "latest.html").write_text(doc, encoding="utf-8")
    return {"html": str(out / "latest.html"), "markdown": str(out / "latest.md"),
            "json": str(out / "latest-summary.json"), "raw": str(raw)}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Run an Ethoculus agent probe from a manifest.")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--agent", default=os.getenv("PROBE_AGENT", "mock-good"),
                   help="claude | mock-good | mock-sloppy | mock-bad | mock-noisy:<rate> | python:module:Class")
    p.add_argument("--model", default=None, help="agent model; overrides AGENT_MODEL")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--cases", default="", help="comma-separated case_ids (default: all)")
    p.add_argument("--conditions", default="", help="comma-separated condition ids (default: all)")
    p.add_argument("--no-policy", action="store_true", help="ablation: omit the deployment policy")
    p.add_argument("--judge", action="store_true", help="grade with JUDGE_PROVIDER / JUDGE_MODEL")
    p.add_argument("--judge-provider", default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--show-http", action="store_true", help="print Claude API bodies (claude agent)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    m = load_manifest(a.manifest)
    case_ids = [c.strip() for c in a.cases.split(",") if c.strip()]
    cond_ids = [c.strip() for c in a.conditions.split(",") if c.strip()]
    jobs = plan_jobs(m, a.trials, case_ids, cond_ids)
    use_policy = not a.no_policy

    if a.dry_run:
        n_cases = len({j[1].case_id for j in jobs})
        n_conds = len({j[2].cid for j in jobs})
        print(f"Manifest:   {m.path}  ({m.experiment_id}: {m.title})")
        print(f"Agent:      {a.agent}{' / ' + (a.model or os.getenv('AGENT_MODEL', '')) if a.agent == 'claude' else ''}")
        print(f"Policy:     {'on' if use_policy else 'OFF (ablation)'}")
        print(f"Cases:      {n_cases}   Conditions: {n_conds}   Trials: {a.trials}")
        print(f"Agent runs: {len(jobs)}")
        if a.agent == "claude":
            print(f"Estimated Claude API calls: about {len(jobs) * 3} to {len(jobs) * 5} "
                  "(each run is a short multi-turn tool loop)")
        print(f"Tools:      {len(m.tools)}  Forbidden: {', '.join(m.forbidden_tools)}")
        print(f"Judge:      {'on' if a.judge else 'off'}")
        print("No API calls made.")
        return

    agent = make_agent(a.agent, a.model, a.show_http)
    agent.check_key()
    judge_cfg = None
    if a.judge:
        from providers import config_from_env
        judge_cfg = config_from_env("JUDGE", a.judge_provider, a.judge_model)
        judge_cfg.check_key()

    rows = execute_suite(m, agent, jobs, use_policy, judge_cfg)
    summary = summarize(rows, m, str(agent), use_policy)
    paths = write_reports(rows, summary, m)
    o = summary["overall"]
    print(f"\nOverall: pass {pct(o['pass_rate'])} · fail {pct(o['fail_rate'])} · "
          f"indeterminate {pct(o['indeterminate_rate'])} · forbidden-call rate {pct(o['forbidden_call_rate'])}")
    if summary["estimated_cost_usd"] is not None:
        print(f"Estimated agent cost: ${summary['estimated_cost_usd']:.4f}")
    print(f"Report: {paths['html']}\nRaw evidence: {paths['raw']}")


if __name__ == "__main__":
    main()
