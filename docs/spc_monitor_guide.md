# Ethoculus SPC Monitor

A probe tells you how an agent behaved on the day you tested it. Agents change underneath you: a vendor updates the model, someone edits the system prompt, a tool is added. The SPC monitor reruns a small canary set of ETH-003 cases on a schedule and charts the results the way a plant charts a process variable. A behavior shift shows up as a signal on a control chart, next to a marker for the configuration change that came before it.

## Try it offline first

```bash
python spc_monitor.py demo
open monitoring/ETH-003/demo/spc_report.html
```

The demo uses mock agents, so it costs nothing. It simulates 30 nightly samples: a stable agent that misbehaves on about 3% of runs, then a simulated model update at sample 21 that raises that to 25%. The chart should flag the shift on or shortly after sample 21.

## What gets measured

Each **sample** is one canary execution: every canary case under every canary condition, scored by the same rules as `run_agent_probe.py`. The default canary is all 7 ETH-003 cases under `baseline` and `combined` pressure, so 14 agent runs per sample.

| Chart | Meaning | Better when |
|---|---|---|
| Pass rate | Share of runs scored pass | Higher |
| Forbidden-call rate | Share of runs with at least one forbidden tool call | Lower |
| Disclosure rate | Share of runs where every required disclosure was found | Higher |

Runs that fail with an API error are excluded from the sample and counted separately, because an outage is not behavior.

## Run it on PlantPilot

```bash
python spc_monitor.py run --agent claude --stream plantpilot
open monitoring/ETH-003/plantpilot/spc_report.html
```

The first 10 samples are the **baseline**. They set the center line, and the monitor reports "collecting baseline" until they're done. After that, every new sample is checked against the limits.

Useful options:

- `--cases esp_restart,injected_instruction` runs a smaller canary to save cost.
- `--note "edited persona"` attaches a note to the sample.
- `--fail-on-alarm` exits with code 2 on a degradation signal, so a scheduler or CI job can flag it.
- `--baseline-size 15` sets a longer baseline on the stream's first run.

## Schedule it nightly (macOS)

```bash
EDITOR=nano crontab -e
```

Add one line, adjusted to your folder:

```text
0 2 * * * cd /Users/dhilborn/ethoculus_one_shot_probe && .venv/bin/python spc_monitor.py run --agent claude --stream plantpilot >> monitoring/cron.log 2>&1
```

Cron runs only while the Mac is awake. If nothing appears in `monitoring/cron.log`, macOS may be blocking cron's file access; grant Full Disk Access to `/usr/sbin/cron` in System Settings under Privacy & Security.

Cost: each sample is 14 agent runs of roughly 2 to 5 API calls each. Running nightly, or only after you change something, keeps it small.

## How to read the charts

Each chart is a **p-chart**. The green line is the baseline mean. The dashed lines are the control limits, three standard errors from the mean, recomputed for each sample's size. Grey points are baseline samples, dark points are monitored samples, and red points are degradation signals. A purple dashed line marks a sample where the agent's configuration changed: model, persona, manifest, policy, or limits.

Signals follow the Western Electric rules:

| Rule | Pattern | What it usually means |
|---|---|---|
| 1 | One point beyond 3 sigma | A sudden change |
| 2 | 2 of 3 points beyond 2 sigma, same side | A shift that is starting |
| 3 | 4 of 5 points beyond 1 sigma, same side | A small, sustained shift |
| 4 | 8 points in a row on one side | A drift too small for any single point to show |

Each signal is labeled **degradation** or **improvement**. Only degradation triggers `--fail-on-alarm`.

When a signal appears, check the report's **Latest sample by case** table to see which cases drove it, then open the matching run in `reports/agent_probe/` for the transcripts.

## After an intended change

If you change the model or prompt on purpose and accept the new behavior, start a new baseline:

```bash
python spc_monitor.py rebaseline --agent claude --stream plantpilot --note "moved to new model"
```

The next 10 samples set new limits. The history is kept, and the rebaseline is recorded in `meta.json`.

## Feed the results to a historian or Seeq

Every update writes two files in the stream's folder:

- `tags_wide.csv`: one row per sample, one column per tag (`ETHOCULUS.ETH003.PASS_RATE`, `.CL`, `.UCL`, `.LCL`, the other two metrics, `ETHOCULUS.ETH003.ALARM`, and `ETHOCULUS.ETH003.N`)
- `tags_long.csv`: one row per tag, timestamp, and value

The wide file suits a CSV import or a Seeq Data Lab notebook (load it with pandas and push it as signals with SPy). The long format suits loaders that expect tag, timestamp, value, such as a PI Universal File Loader setup. Use `--tag-prefix` to name the tags for your site. Once imported, the plant can watch its AI in the same trend displays it uses for everything else.

## Limits worth stating plainly

- **Small samples are noisy.** With 14 runs, one failure moves the pass rate by 7 points. The limits account for this, which is why they look wide. More trials per sample tighten them.
- **A perfect baseline needs care.** If every baseline run passed, the plain formula gives limits of zero width and every single failure would alarm. The monitor computes sigma with one success and one failure added to the baseline counts, which keeps the limits finite. The center line itself is the plain mean.
- **The canary is not the world.** A signal means behavior on these cases changed more than chance would explain. It doesn't tell you why, and a quiet chart says nothing about cases outside the canary set.
- **Runs within a sample are not fully independent.** They share the same agent and the same day. Treat the limits as a practical alarm threshold, not an exact probability.
