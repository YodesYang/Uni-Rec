#!/usr/bin/env python3
"""Generate T30 quantile-based time_bucket_boundaries (63 values).

Based on EXP-048 train EDA q5_time_diff data.

Strategy (resident researcher view):
- Use seq_d quantiles as primary anchor (events/day=143 · session-level
  immediate intent · most LB-relevant).
- 但 seq_d p99=56d doesn't cover seq_a/seq_b/seq_c long tails (up to 22mo).
- Solution: 63 boundaries with denser low-end (seconds to minutes for fast
  recency) + seq_d quantile fits (15min ~ 56d) + extension for rare long tails.

Hybrid plan:
- 12 boundaries: 1s ~ 1min (default-like fast recency · keep)
- 51 boundaries: log-spaced from 1min to seq_d p99 (~56d)
  using seq_d quantile distribution.
"""
import numpy as np

# seq_d quantiles from EXP-048
# Key observations: events/day ≈ 143 (high density) · p1=34652s ≈ 9.6h ·
# p50=1636691s ≈ 19d · p99=4828126s ≈ 56d
# Default 0~3600 (1h) covers seq_d p1 ~ p1 · so default low-end OK
# Default 86400 (1d) ≈ seq_d p25 (8.3d worth) · need denser middle/high

# Strategy: hybrid
# - 12 fixed low-end boundaries (1s ~ 1min granularity for fast recency)
# - 51 log-quantile-derived boundaries from 1min to ~60d max

# Step 1 · low-end fixed (12 boundaries, capture "in-page recency" 1s-60s)
LOW_END = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
# Step 2 · seq_d-anchored log-quantile from 60s to 56d (51 boundaries)
# Use np.geomspace to create log-spaced values in [60, p99=4828126]
SEQ_D_P99 = 4_828_126  # seconds (~ 56 days)
SEQ_D_P50 = 1_636_691  # seconds (~ 19 days)
mid_high = np.geomspace(60, SEQ_D_P99, 51).astype(np.int64)  # 51 values from 60 to 56d
# But we already have 60 in LOW_END · drop the first
mid_high = mid_high[1:]  # 50 values, starting just above 60

# Total: 12 (low) + 50 (mid_high) = 62 · need 63
# Add 1 more anchor for seq_c long tail (cover p99 of seq_c ≈ 54M sec = 22 months)
# Append a single late-tail boundary
LATE_TAIL = [54_354_848]  # seq_c p99 (covers lifetime anchor)
# Total: 12 + 50 + 1 = 63 ✅

bounds = LOW_END + list(mid_high) + LATE_TAIL
bounds_np = np.array(sorted(set(bounds)), dtype=np.int64)

# Verify constraints
assert len(bounds_np) == 63, f"Got {len(bounds_np)} boundaries, expected 63"
assert (bounds_np > 0).all(), "All boundaries must be positive"
assert (bounds_np[1:] > bounds_np[:-1]).all(), "Must be strictly increasing"

# Sanity vs default
DEFAULT = [
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
]

print("=" * 70)
print("T30 Quantile-based boundaries (63 values)")
print("=" * 70)
print(f"Length: {len(bounds_np)}")
print(f"Min: {bounds_np[0]}s")
print(f"Max: {bounds_np[-1]}s = {bounds_np[-1] / 86400:.1f} days")
print()
print("Boundaries (seconds):")
print(",".join(str(b) for b in bounds_np))
print()
print("Boundaries (days for >1d values):")
for b in bounds_np:
    if b >= 86400:
        print(f"  {b}s = {b / 86400:.1f}d")

# Compare with default
print()
print("=" * 70)
print("Comparison with DEFAULT")
print("=" * 70)
print(f"Default range: {DEFAULT[0]} ~ {DEFAULT[-1]}s ({DEFAULT[-1] / 86400:.0f} days)")
print(f"New range:    {bounds_np[0]} ~ {bounds_np[-1]}s ({bounds_np[-1] / 86400:.0f} days)")

# How many boundaries are in 0-3600 (1h) range
def_low = sum(1 for b in DEFAULT if b <= 3600)
new_low = sum(1 for b in bounds_np if b <= 3600)
print(f"Default 0-1h:  {def_low}/63 = {def_low/63*100:.0f}%")
print(f"New 0-1h:     {new_low}/63 = {new_low/63*100:.0f}%")

def_day = sum(1 for b in DEFAULT if b <= 86400)
new_day = sum(1 for b in bounds_np if b <= 86400)
print(f"Default 0-1d:  {def_day}/63 = {def_day/63*100:.0f}%")
print(f"New 0-1d:     {new_day}/63 = {new_day/63*100:.0f}%")
