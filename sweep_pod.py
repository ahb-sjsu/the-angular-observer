"""NRP shard worker — one CPU pod processes a slice of the draw space.

Indexed Job: JOB_COMPLETION_INDEX picks the slice; results printed to stdout as
JSON between markers for log-based aggregation (no PVC needed). Single process
(pod is cpu=1); BLAS pinned to 1 thread via sweep.py's import-time env set.
"""

import os
import json
import math
import numpy as np
from sweep import worker  # reuses gen+score+SIGALRM-timeout logic

IDX = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
N_SHARDS = int(os.environ.get("N_SHARDS", "1"))
DRAWS_TOTAL = int(os.environ.get("DRAWS_TOTAL", "20000"))
TOPK = int(os.environ.get("TOPK", "50"))

per = math.ceil(DRAWS_TOTAL / N_SHARDS)
start = IDX * per
end = min(start + per, DRAWS_TOTAL)

print(f"shard {IDX}/{N_SHARDS}: draws [{start},{end})", flush=True)
results = []
for i in range(start, end):
    m = worker(i)
    if m is not None:
        results.append(m)

results.sort(
    key=lambda d: d["score"] if np.isfinite(d["score"]) else -1e9, reverse=True
)
top = results[:TOPK]
print(f"shard {IDX}: {len(results)} scored, emitting top {len(top)}", flush=True)
print("###RESULTS_JSON_START###", flush=True)
print(json.dumps(top), flush=True)
print("###RESULTS_JSON_END###", flush=True)
