"""Quick smoke for Q6/Q7 CVR-by-time bucketing in eda_quantile.py.

No parquet required - synthesize timestamp + label arrays and verify
the bucket aggregation matches expected counts/CVR.
"""
import numpy as np

# Simulate 1000 rows: 600 on 2026-03-03 (元宵节 · CVR 0.4) + 400 on
# 2026-03-04 (节后翌日 · CVR 0.2), spread across 24h UTC.
np.random.seed(42)
n_03_03 = 600
n_03_04 = 400

# UTC days since epoch
day_03_03 = 20515  # 2026-03-03 UTC (元宵节)
day_03_04 = 20516  # 2026-03-04 UTC (元宵节翌日)

# Timestamps: random hours within each day, full 24h spread
ts_03_03 = day_03_03 * 86400 + np.random.randint(0, 86400, n_03_03)
ts_03_04 = day_03_04 * 86400 + np.random.randint(0, 86400, n_03_04)
ref_ts = np.concatenate([ts_03_03, ts_03_04]).astype(np.int64)

# Labels (binary): 03-03 CVR 0.4 / 03-04 CVR 0.2
labels = np.concatenate([
    (np.random.rand(n_03_03) < 0.4).astype(np.int64),
    (np.random.rand(n_03_04) < 0.2).astype(np.int64),
])

# === Q6 logic (mirrors eda_quantile.py L367-382) ===
cvr_by_day = {}
day_idx = (ref_ts.astype(np.int64) // 86400).astype(np.int64)
day_min = int(day_idx.min())
day_max = int(day_idx.max())
print(f"day_min={day_min} day_max={day_max} (expected 20517-20518)")
for d in range(day_min, day_max + 1):
    mask = (day_idx == d)
    cnt = int(mask.sum())
    if cnt == 0:
        continue
    pos = int(labels[mask].sum())
    cvr_by_day[d] = [cnt, pos]

# Verify
for d, (cnt, pos) in cvr_by_day.items():
    cvr = pos / cnt
    print(f"  day={d} cnt={cnt} pos={pos} cvr={cvr:.4f}")
expected_03_03 = cvr_by_day[day_03_03]
expected_03_04 = cvr_by_day[day_03_04]
assert expected_03_03[0] == 600
assert expected_03_04[0] == 400
# CVR ~0.4 / ~0.2 ± 0.05 (noise)
assert 0.35 < expected_03_03[1] / 600 < 0.45
assert 0.15 < expected_03_04[1] / 400 < 0.25
print("\n[OK] Q6 day bucketing correct\n")

# === Q7 logic (mirrors eda_quantile.py L384-395) ===
cvr_by_hour = {}
hour_of_day = ((ref_ts.astype(np.int64) // 3600) % 24).astype(np.int64)
for h in range(24):
    mask = (hour_of_day == h)
    cnt = int(mask.sum())
    if cnt == 0:
        continue
    pos = int(labels[mask].sum())
    cvr_by_hour[h] = [cnt, pos]

# Verify all 24 hours represented (with random uniform spread)
total_cnt = sum(v[0] for v in cvr_by_hour.values())
assert total_cnt == 1000, f"hour total {total_cnt} != 1000"
# All 24 hours should have some samples (probabilistically · n=1000 / 24 ≈ 42 per hour)
assert len(cvr_by_hour) == 24, f"covered hours {len(cvr_by_hour)} != 24"
# 北京 = UTC + 8 mod 24
print("# Hour CVR (UTC -> 北京 hour):")
for h in sorted(cvr_by_hour.keys())[:6]:  # first 6 hours sample
    cnt, pos = cvr_by_hour[h]
    bj_h = (h + 8) % 24
    print(f"  UTC {h:02d} -> 北京 {bj_h:02d} cnt={cnt} cvr={pos/cnt:.4f}")
print("\n[OK] Q7 hour bucketing correct (24 hours covered)\n")

# === Date string formatting (mirrors _cvr_bucket_summary L505-510) ===
from datetime import datetime, timezone, timedelta
ts_utc = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_03_03)
ts_beijing = ts_utc + timedelta(hours=8)
print(f"# Date conversion: UTC days {day_03_03} -> {ts_utc.strftime('%Y-%m-%d')} (UTC) "
      f"/ {ts_beijing.strftime('%Y-%m-%d')} (北京) · weekday={ts_utc.weekday()}")
assert ts_utc.strftime('%Y-%m-%d') == '2026-03-03', \
       f"day {day_03_03} should be 2026-03-03 UTC · got {ts_utc.strftime('%Y-%m-%d')}"
print("\n[OK] Date formatting correct\n")

print("All Q6/Q7 smoke tests passed.")
