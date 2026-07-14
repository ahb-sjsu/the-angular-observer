"""Runs ON ATLAS. Pulls each wob-3axl pod's log, merges the JSON blocks, and
scores the PRE-REGISTERED rung 3a-XL analysis (PREREG_RUNG3.md):

  universality bar  >= 80% of gated qualifying rules have mean consecutive
                    angle stability >= 0.9 x their own mean consecutive churn
  validity gate     every consecutive churn value >= 0.5 (else the rule is
                    substrate-unstable: excluded from the denominator, counted;
                    if > 30% fail the gate, that is the headline instead)
  law               regression of long-lever angle stability on emergent dim
                    (slope + Spearman), reported alongside the fidelity law
"""

import json
import subprocess

import numpy as np
from scipy.stats import spearmanr

NS = "ssu-atlas-ai"
S, E = "###RESULTS_JSON_START###", "###RESULTS_JSON_END###"


def kc(*args):
    return subprocess.run(
        ["kubectl", "-n", NS, *args], capture_output=True, text=True
    ).stdout


pods = [
    p for p in kc("get", "pods", "-l", "job-name=wob-3axl", "-o", "name").split() if p
]
merged, ok, bad = [], 0, 0
for p in pods:
    log = kc("logs", p)
    if S in log and E in log:
        try:
            merged.extend(json.loads(log.split(S)[1].split(E)[0].strip()))
            ok += 1
        except Exception:
            bad += 1
    else:
        bad += 1

seen, rules = set(), []
for m in merged:
    key = (str(m["lhs"]), str(m["rhs"]))
    if key not in seen:
        seen.add(key)
        rules.append(m)

with open("/home/claude/wob_3axl/merged_3axl.json", "w") as f:
    json.dump(rules, f, indent=1)

measured = [r for r in rules if r.get("stability")]
stalled = [r for r in rules if not r.get("stability")]
gated, unstable = [], []
for r in measured:
    st = r["stability"]
    (gated if min(st["churn"]) >= 0.5 else unstable).append(r)

print(
    f"pods={len(pods)} parsed_ok={ok} bad={bad} | qualifying rules: "
    f"{len(rules)} unique, {len(measured)} measured, {len(stalled)} stalled"
)
print(
    f"validity gate (all consec churn >= 0.5): {len(gated)} pass, "
    f"{len(unstable)} substrate-unstable "
    f"({100 * len(unstable) / max(len(measured), 1):.0f}% — "
    f"headline flips if > 30%)\n"
)

print(
    f"  {'dim':>5} {'rho':>5} {'fid_last':>8} {'consec':>7} {'churn':>6} "
    f"{'long_a':>7} {'long_g':>7}   rule"
)
hits = 0
for r in sorted(gated, key=lambda x: x["dim"]):
    st = r["stability"]
    ca, cg = np.mean(st["consec"]), np.mean(st["churn"])
    hit = ca >= 0.9 * cg
    hits += hit
    print(
        f"  {r['dim']:5.2f} {r['angle_rho']:5.2f} {st['fid'][-1]:8.2f} "
        f"{ca:7.2f} {cg:6.2f} {st['long_angle']:7.2f} "
        f"{st['long_churn']:7.2f}   "
        f"{'' if hit else '<- MISS  '}{r['lhs']} -> {r['rhs']}"
    )

if gated:
    frac = hits / len(gated)
    print(
        f"\nuniversality: {hits}/{len(gated)} = {100 * frac:.0f}% meet "
        f"consec >= 0.9 x own churn  (bar: >= 80%) -> "
        f"{'PASS' if frac >= 0.8 else 'FAIL'}"
    )
    dims = np.array([r["dim"] for r in gated])
    longs = np.array([r["stability"]["long_angle"] for r in gated])
    if len(gated) >= 5 and np.ptp(dims) > 0.2:
        slope = np.polyfit(dims, longs, 1)[0]
        rho = spearmanr(dims, longs).statistic
        print(
            f"law: long-lever stability vs dim — slope {slope:+.3f}/dim, "
            f"Spearman {rho:+.3f}, n={len(gated)} "
            f"(fidelity law reference: −0.157/dim, Spearman −0.881)"
        )
    else:
        print("law: too few gated rules / dim range too narrow for a slope")
