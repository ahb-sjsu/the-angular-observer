"""Runs ON ATLAS. Pulls each wob-sweep pod's log, extracts the JSON block
between the markers, merges + ranks globally, writes merged + prints top rules."""
import json
import math
import subprocess

NS = "ssu-atlas-ai"
S, E = "###RESULTS_JSON_START###", "###RESULTS_JSON_END###"


def kc(*args):
    return subprocess.run(["kubectl", "-n", NS, *args],
                          capture_output=True, text=True).stdout


pods = [p for p in kc("get", "pods", "-l", "job-name=wob-sweep",
                      "-o", "name").split() if p]
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

merged.sort(key=lambda d: d["score"] if math.isfinite(d.get("score", float("nan")))
            else -1e9, reverse=True)
with open("/home/claude/wolfram-observer-bridge/merged_results.json", "w") as f:
    json.dump(merged, f, indent=2)

print(f"pods={len(pods)} parsed_ok={ok} bad={bad} total_rules={len(merged)}\n")
print(f"  {'score':>6} {'angle_rho':>9} {'spread':>7} {'dim':>5} {'N':>5}   rule")
seen = set()
shown = 0
for m in merged:
    key = (str(m["lhs"]), str(m["rhs"]))
    if key in seen:
        continue
    seen.add(key)
    print(f"  {m['score']:6.3f} {m['angle_rho']:9.3f} {m['spread']:7.2f} "
          f"{m['dim']:5.2f} {m['N']:5d}   {m['lhs']} -> {m['rhs']}")
    shown += 1
    if shown >= 25:
        break

# how many clear the manifold bar
mani = [m for m in merged if m["angle_rho"] >= 0.80 and m["spread"] <= 0.30]
hi_d = [m for m in mani if m["dim"] >= 2.5]
print(f"\nmanifold-grade rules (angle_rho>=0.80 & spread<=0.30): {len(mani)}")
print(f"  of which dim>=2.5 (2D+ spaces): {len(hi_d)}")
