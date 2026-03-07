# Dashboard parity checklist: local vs external (EC2)

Use this to ensure the external Streamlit dashboard shows the **same metrics and charts** as the local one.

**Quick check:** Run the snapshot script on both environments and diff the JSON files (see Section 4). If they match, the data driving all metrics and charts is the same.

---

## 1. Data source (must match)

| Item | Source | Local | External | Notes |
|------|--------|--------|----------|--------|
| **Parquet file** | First existing path in `settings.parquet_paths` | `data/processed/dashboard_summaries/donors_dashboard_slim.parquet` (if present) | Same path relative to project root | If EC2 uses a different path (e.g. only full parquet), row counts and chart data will differ. |
| **Row count** | Loaded DataFrame | ___ | ___ | Run the snapshot script (see below) on both; `data_source.rows` must match. |
| **training_summary.json** | `models/donor_model_checkpoints/training_summary.json` | Present | Present | Required for AUC, F1, baseline_auc, lift when `USE_SAVED_METRICS_ONLY = True`. |

**Action:** On EC2, ensure the **same** slim parquet and `training_summary.json` are deployed (same files as local, or generated from the same pipeline).

---

## 2. Configuration (must match)

| Setting | File | Value | Notes |
|---------|------|--------|--------|
| **USE_SAVED_METRICS_ONLY** | `dashboard/config/settings.py` | `True` | When True, AUC/F1 etc. come only from `training_summary.json`. |
| **SAVED_METRICS_CANDIDATES** | Same | First path = `models/donor_model_checkpoints/training_summary.json` | Resolved relative to project root. |
| **parquet_paths order** | Same | Slim parquet first | Loader uses first **existing** path. |

**Action:** Do not change these between local and EC2. Same codebase = same config.

---

## 3. Metrics inventory (all must match)

### Sidebar (every page)

| Metric | Data source | Expected |
|--------|-------------|----------|
| AUC Score | `get_model_metrics()` → saved or parquet | e.g. 94.89% |
| F1 Score | Same | e.g. 85.34% |
| Baseline AUC | Same | e.g. 50.29% or from JSON |
| Lift vs Baseline | Same | e.g. +88.8% or from JSON |

### Executive Summary page

| Metric | Data source | Expected |
|--------|-------------|----------|
| High Confidence Prospects (>70%) | Count from `df` where `Will_Give_Again_Probability >= 0.7` | Integer; same as local if same parquet |
| Untapped Donor Potential ($) | `avg_gift * high_prob_count * conversion` from `df` | Same formula and data |
| **Hero card 1: AUC Score** | `get_model_metrics()` | Same as sidebar |
| **Hero card 2: F1 Score** | Same | Same as sidebar |
| **Hero card 3: Revenue Potential** | From `df` (high prob, avg_gift, outcome) | Same as “Untapped” logic |
| **Hero card 4: Improvement** | Lift or (AUC − baseline) / baseline | Same as sidebar lift |

### Charts – Executive Summary

| Chart | Data source | Must match |
|-------|-------------|------------|
| Donor Base Breakdown (segment bars) | `df`: groupby `segment`, count + avg probability | Same segment counts and percentages |
| Donors Predicted to Give in 2025 (tiers) | `df`: `pd.cut(prob, [0, 0.4, 0.7, 1.0])` → Low/Medium/High counts | Same tier counts |
| Portfolio Quality by Gift Officer | `df`: groupby `Primary_Manager`, median prob, recency | Same if same parquet and columns |

### Model Comparison page

| Metric / chart | Data source | Must match |
|----------------|-------------|------------|
| Fusion AUC, F1, Accuracy, Precision, Recall, Specificity | `get_model_metrics()` + optional compute from `df` | Same as sidebar when saved metrics used |
| Baseline metrics | Same + baseline from `days_since_last` | Same |
| Bar chart (Fusion vs Baseline) | Same metrics | Same numbers |
| Radar chart | Same metrics | Same values |

### Performance page

| Metric / chart | Data source | Must match |
|----------------|-------------|------------|
| AUC, F1, Accuracy cards | `get_model_metrics()` | Same as sidebar |
| ROC curve | `df`: `outcome_col`, `prob_col` | Same curve if same data |
| Precision–Recall curve | Same | Same |
| Confusion matrix | Same + threshold from saved `optimal_threshold` | Same |

### Business Impact page

| Metric / chart | Data source | Must match |
|----------------|-------------|------------|
| Revenue / ROI numbers | `df`: high_prob donors, avg_gift, conversion | Same parquet → same numbers |
| Revenue by segment bar chart | `df` groupby segment | Same |
| ROI by segment | Same | Same |

### Features page

| Chart | Data source | Must match |
|-------|-------------|------------|
| Feature importance | Correlation with outcome from `df` | Same if same columns and outcome |
| Box plots / distributions | `df` columns | Same data |

### Take Action page

| Content | Data source | Must match |
|---------|-------------|------------|
| Table / list of donors | Filtered `df` | Same filters and parquet → same rows |

---

## 4. How to verify parity

### Step A: Export snapshots

**Local (from project root, with venv active):**
```bash
cd LMUCapstoneProject
python -m dashboard.export_dashboard_snapshot --output local_snapshot.json
```

**External (on EC2, same repo and venv):**
```bash
cd ~/capstone-app   # or your app root
python -m dashboard.export_dashboard_snapshot --output external_snapshot.json
```
Then copy `external_snapshot.json` to your machine (e.g. SCP, WinSCP).

### Step B: Compare snapshots

- **Diff the two JSON files** (e.g. `diff local_snapshot.json external_snapshot.json`, or a JSON diff tool).
- Check in particular:
  - `data_source.parquet_path` → same file used (or at least same row count).
  - `data_source.rows` → same.
  - `saved_metrics` → both present and identical (or both null if you don’t use saved metrics).
  - `model_metrics.auc`, `model_metrics.f1` → same.
  - `executive_summary.high_confidence_count`, `estimated_revenue`, `revenue_display` → same.
  - `charts.segment_breakdown`, `charts.tier_counts` → same.

### Step C: Visual check (optional)

1. Open the dashboard **locally** and note: sidebar AUC/F1, Executive Summary hero metrics, segment and tier chart numbers.
2. Open the **external** URL and compare the same numbers and charts.
3. Repeat for Model Comparison, Performance, Business Impact if you care about full parity.

---

## 5. Common causes of mismatch

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| AUC/F1 = N/A on external only | `training_summary.json` not found on EC2 | Copy file to `models/donor_model_checkpoints/` on EC2; ensure path is correct for `get_project_root()`. |
| Different row count | Different parquet file (e.g. full vs slim) | Deploy same slim parquet to EC2; ensure it’s first in `parquet_paths` that exists. |
| Different segment/tier counts | Different parquet or different columns | Use same slim parquet; ensure precompute script and column names match. |
| Different revenue / high-confidence | Different data or missing columns | Same parquet and same `avg_gift` / outcome columns (e.g. `Gave_Again_In_2025`). |
| Sidebar correct, Executive Summary wrong | Old cache or duplicate metric logic | Restart Streamlit; ensure latest code (no duplicate metric calculation on Executive Summary). |

---

## 6. One-line parity check (after snapshots)

```bash
# After exporting local_snapshot.json and external_snapshot.json
diff <(jq -S . local_snapshot.json) <(jq -S . external_snapshot.json)
```

If the diff is empty (and both use the same parquet and saved metrics), the backend data driving metrics and charts is the same. Any remaining UI differences would be from layout or client-side behavior, not from metrics or chart data.
