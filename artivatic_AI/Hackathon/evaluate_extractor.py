#!/usr/bin/env python3
"""
Hospital Bill Extractor — Evaluation & Metrics Dashboard  v5
=============================================================
All v4 logic retained (FIX A–G).

NEW — GPU COST SECTION (FIX H)
  • Reads real gpu_cost_inr + processing_time_s from every output JSON _meta
  • Computes: per-file, total-batch, per-1000, annual projected costs
  • Prints a dedicated  ▌ GPU COST METRICS  section in the report
  • Adds gpu_cost block to _evaluation_report_v5.json
  • Constants at top: GPU_NAME, GPU_TDP_WATTS, GPU_UTIL_PCT, SYSTEM_WATTS,
    ELEC_RATE_INR — used when _meta.gpu_cost_inr is absent (v10 or older files)
"""

from __future__ import annotations

import ast
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
BASE             = Path(r"C:\Users\ravip\OneDrive\Desktop\Artivatic_AI_Hackathon\artivatic_ai\hackathon")
OUT_DIR          = BASE / "output"
GROUND_TRUTH     = BASE / "ground_truth"
EXTRACTOR_SCRIPT = Path(__file__).parent / "hospital_bill_extractor_v11.py"

# Amount tolerance
AMOUNT_TOL_FLAT = 2.0
AMOUNT_TOL_PCT  = 0.005

# Jaccard threshold
JACCARD_THRESHOLD = 0.4

# ── GPU COST CONFIGURATION — must match extractor settings ───────────────────
GPU_NAME      = "RTX 3060"
GPU_TDP_WATTS = 170
GPU_UTIL_PCT  = 0.40
SYSTEM_WATTS  = 80
ELEC_RATE_INR = 6.5
# ══════════════════════════════════════════════════════════════════════════════

EVAL_FIELDS = [
    "bill_type", "bill_number", "bill_date",
    "hospital.name", "hospital.gstin",
    "patient.name", "patient.age", "patient.gender",
    "admission.admission_date", "admission.discharge_date",
    "admission.los_days", "admission.ward",
    "clinical.attending_doctor",
    "charges_summary.gross_total", "charges_summary.balance_due",
    "charges_summary.discount", "charges_summary.advance_paid",
    "insurance.cashless", "insurance.policy_number",
]

NUMERIC_FIELDS = {
    "charges_summary.gross_total", "charges_summary.balance_due",
    "charges_summary.discount", "charges_summary.advance_paid",
    "admission.los_days", "patient.age",
}

DATE_FIELDS = {"bill_date", "admission.admission_date", "admission.discharge_date"}

FUZZY_STRING_FIELDS = {
    "hospital.name", "patient.name", "clinical.attending_doctor",
    "admission.ward", "bill_type",
}


# ══════════════════════════════════════════════════════════════════════════════
#  GPU COST ENGINE  (mirrors extractor — keep in sync)
# ══════════════════════════════════════════════════════════════════════════════

def _gpu_effective_watts() -> float:
    """Return actual power draw: GPU active draw + system overhead."""
    return GPU_TDP_WATTS * GPU_UTIL_PCT + SYSTEM_WATTS


def _gpu_cost_for_seconds(seconds: float) -> float:
    """Calculate electricity cost in ₹ for a number of seconds of GPU use."""
    kw    = _gpu_effective_watts() / 1000.0
    hours = seconds / 3600.0
    return kw * hours * ELEC_RATE_INR


def _build_gpu_summary(
    total_seconds: float,
    n_files:       int,
    total_cost_inr: float,
) -> dict:
    """
    Build GPU cost summary dict from real measured costs.

    total_cost_inr: sum of per-file gpu_cost_inr from JSON _meta (preferred).
    Falls back to formula if 0 (older extractor versions without _meta field).
    """
    if total_cost_inr <= 0:
        total_cost_inr = _gpu_cost_for_seconds(total_seconds)

    per_file    = total_cost_inr / n_files if n_files else 0.0
    per_1000    = per_file * 1000
    annual_cost = per_file * n_files * 22 * 12

    return {
        "gpu_name":              GPU_NAME,
        "effective_watts":       round(_gpu_effective_watts(), 1),
        "elec_rate_inr_per_kwh": ELEC_RATE_INR,
        "total_seconds":         round(total_seconds, 2),
        "n_files":               n_files,
        "cost_per_file_inr":     round(per_file, 6),
        "total_cost_inr":        round(total_cost_inr, 6),
        "cost_per_1000_inr":     round(per_1000, 4),
        "annual_cost_inr":       round(annual_cost, 2),
        "gpt4v_equivalent_usd":  round(n_files * 0.03, 2),
        "savings_pct":           round(
            (1 - total_cost_inr / max(n_files * 0.03 * 83, 0.0001)) * 100, 1
        ),
    }


def print_gpu_cost_section(gpu: dict, width: int = 72) -> None:
    """Print the ▌ GPU COST METRICS section in evaluator report style."""
    dot = lambda lbl, val, w=44: f"  {lbl}{'.' * max(1, w - len(lbl))} {val}"
    hr  = "─" * width

    print(f"\n\n{'▌ GPU COST METRICS':^{width}}")
    print(hr)
    print(dot("GPU",                        f"{gpu['gpu_name']}  ({gpu['effective_watts']}W effective)"))
    print(dot("Electricity rate",           f"₹{gpu['elec_rate_inr_per_kwh']}/kWh"))
    print(dot("Total OCR time (GPU)",       f"{gpu['total_seconds']:.2f}s across {gpu['n_files']} files"))
    print("·" * width)
    print(dot("Cost per file",              f"₹{gpu['cost_per_file_inr']:.6f}"))
    print(dot("Total batch cost",           f"₹{gpu['total_cost_inr']:.6f}"))
    print(dot("Cost per 1,000 bills",       f"₹{gpu['cost_per_1000_inr']:.4f}"))
    print(dot("Projected annual cost",
              f"₹{gpu['annual_cost_inr']:.2f}  (~${gpu['annual_cost_inr']/83:.2f} USD)"))
    print(dot("GPT-4V cloud equivalent",    f"~${gpu['gpt4v_equivalent_usd']:.2f}  "
                                             f"(₹{gpu['gpt4v_equivalent_usd']*83:.0f})"))
    print(dot("Savings vs GPT-4V cloud",    f"{gpu['savings_pct']:.1f}%  — local GPU is essentially free"))

    # Mini bar
    savings_bar_w = 28
    filled = round(min(gpu["savings_pct"], 100) / 100 * savings_bar_w)
    bar    = "█" * filled + "░" * (savings_bar_w - filled)
    print(f"\n  Savings  {bar}  {gpu['savings_pct']:.1f}%")
    print(f"\n  Cost Rating: 🏆 Effectively $0 — local GPU OCR beats cloud by {gpu['savings_pct']:.0f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION LOGIC  (unchanged from v4)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FieldResult:
    """Outcome of evaluating one field in one document."""
    field:     str
    extracted: Any
    expected:  Any
    outcome:   str


@dataclass
class DocumentMetrics:
    """Per-document evaluation results."""
    filename:      str
    processing_s:  float
    field_results: list[FieldResult] = field(default_factory=list)

    @property
    def tp(self) -> int:  return sum(1 for r in self.field_results if r.outcome == "TP")
    @property
    def tn(self) -> int:  return sum(1 for r in self.field_results if r.outcome == "TN")
    @property
    def fp(self) -> int:  return sum(1 for r in self.field_results if r.outcome == "FP")
    @property
    def fn(self) -> int:  return sum(1 for r in self.field_results if r.outcome == "FN")

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0
    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total else 0.0


_MISSING = object()


def _get(obj: dict, dotpath: str) -> Any:
    """Traverse nested dict with dot-notation key."""
    for k in dotpath.split("."):
        if not isinstance(obj, dict) or k not in obj:
            return _MISSING
        obj = obj[k]
    return obj


def _is_empty(v: Any) -> bool:
    if v is None or v is _MISSING: return True
    if isinstance(v, str) and v.strip() == "": return True
    if isinstance(v, (list, dict)) and len(v) == 0: return True
    return False


def _tokenise(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenise(a), _tokenise(b)
    if not ta and not tb: return 1.0
    if not ta or not tb:  return 0.0
    return len(ta & tb) / len(ta | tb)


_DR_PREFIX  = re.compile(r"^(?:Dr\.?|DR\.?|Prof\.?|Prof\s+)\s*", re.I)
_HOSP_NOISE = re.compile(
    r"\s*(?:Healthcare\s+(?:Centre|Center)|Hospital|Clinic|Medical\s+(?:Centre|Center)|"
    r"Ltd\.?|Pvt\.?|Private\s+Limited|Super\s+Speciality|Multi\s+Speciality)\s*$", re.I,
)


def _norm_doctor(s: str) -> str:
    return re.sub(r"\s+", " ", _DR_PREFIX.sub("", s.strip())).strip().lower()


def _norm_hospital(s: str) -> str:
    return re.sub(r"\s+", " ", _HOSP_NOISE.sub("", s.strip())).strip().lower()


def _norm_str(s: Any, field: str = "") -> str:
    if _is_empty(s): return ""
    raw = re.sub(r"\s+", " ", str(s).strip())
    if field == "clinical.attending_doctor": return _norm_doctor(raw)
    if field == "hospital.name":             return _norm_hospital(raw)
    return raw.lower().strip(" .,;:-|/\\")


def _parse_numeric(v: Any) -> float | None:
    if v is None or v is _MISSING: return None
    if isinstance(v, (int, float)): return float(v)
    s = re.sub(r"[,\s₹]", "", str(v)).rstrip(".")
    try: return float(s)
    except ValueError: return None


def _normalise_date(v: Any) -> str:
    if _is_empty(v): return ""
    s = str(v).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        y = int(s[:4])
        return s if 2000 <= y <= 2035 else ""
    m = re.match(r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})$", s)
    if m:
        y = int(m.group(3))
        if 2000 <= y <= 2035:
            return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
    m = re.search(r"(\d{4})", s)
    if m and len(s) >= 8: return s[:10]
    return s


def _amounts_match(ext: Any, exp: Any) -> bool:
    e_num = _parse_numeric(ext)
    g_num = _parse_numeric(exp)
    if e_num is None or g_num is None: return False
    tol = max(AMOUNT_TOL_FLAT, abs(g_num) * AMOUNT_TOL_PCT)
    return abs(e_num - g_num) <= tol


def _values_match(extracted: Any, expected: Any,
                  numeric: bool = False, is_date: bool = False, field: str = "") -> bool:
    if is_date:   return _normalise_date(extracted) == _normalise_date(expected)
    if numeric:   return _amounts_match(extracted, expected)
    if isinstance(expected, bool) or isinstance(extracted, bool):
        return bool(extracted) == bool(expected)
    e_str = _norm_str(extracted, field)
    g_str = _norm_str(expected, field)
    if not e_str and not g_str: return True
    if not e_str or not g_str:  return False
    if e_str == g_str:          return True
    if field in FUZZY_STRING_FIELDS:
        return _jaccard(e_str, g_str) >= JACCARD_THRESHOLD
    return False


def _classify_outcome(extracted: Any, expected: Any,
                      numeric: bool = False, is_date: bool = False, field: str = "") -> str:
    ext_empty = _is_empty(extracted)
    exp_empty = _is_empty(expected)
    if exp_empty and ext_empty:      return "TN"
    if exp_empty and not ext_empty:  return "FP"
    if not exp_empty and ext_empty:  return "FN"
    return "TP" if _values_match(extracted, expected, numeric, is_date, field) else "FP"


def _clean_item_desc(s: str) -> str:
    s = re.sub(r"^\$?SER[A-Z0-9]{3,12}\s+", "", s.strip(), flags=re.I)
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()[:30]


def _item_amounts(item: dict) -> list[float]:
    vals = []
    for k in ("amount", "exc_amount", "total_rs", "unit_rate"):
        v = _parse_numeric(item.get(k))
        if v is not None and v > 0:
            vals.append(v)
    return vals


def _amt_close(a: float, b: float) -> bool:
    tol = max(2.0, max(abs(a), abs(b)) * 0.01)
    return abs(a - b) <= tol


def _score_pair(ext: dict, gt: dict) -> float:
    ec = _clean_item_desc(ext.get("description", ""))
    gc = _clean_item_desc(gt.get("description", ""))
    ea = _item_amounts(ext)
    ga = _item_amounts(gt)
    desc_ok = bool(ec and gc and _jaccard(ec, gc) >= 0.35)
    amt_ok  = bool(ea and ga and any(_amt_close(e, g) for e in ea for g in ga))
    if desc_ok and amt_ok: return 1.0
    if desc_ok:            return 0.7
    if amt_ok:             return 0.4
    return 0.0


def _compare_line_items(extracted_items: list[dict], gt_items: list[dict]) -> dict:
    """FIX E: Best-of-N greedy matching sorted by score descending."""
    if not gt_items: return {}
    triples: list[tuple[float, int, int]] = []
    for ei, ext in enumerate(extracted_items):
        for gi, gt_ in enumerate(gt_items):
            score = _score_pair(ext, gt_)
            if score > 0:
                triples.append((score, ei, gi))
    triples.sort(key=lambda x: -x[0])
    gt_matched:  dict[int, float] = {}
    ext_matched: set[int]         = set()
    for score, ei, gi in triples:
        if ei in ext_matched or gi in gt_matched: continue
        gt_matched[gi]  = score
        ext_matched.add(ei)
    total_credit = sum(gt_matched.values())
    gt_count     = len(gt_items)
    ext_count    = len(extracted_items)
    return {
        "gt_count":          gt_count,
        "extracted_count":   ext_count,
        "matched_full":      sum(1 for c in gt_matched.values() if c >= 1.0),
        "matched_desc_only": sum(1 for c in gt_matched.values() if 0.65 <= c < 1.0),
        "matched_amt_only":  sum(1 for c in gt_matched.values() if c < 0.65),
        "missed":            gt_count - len(gt_matched),
        "spurious":          ext_count - len(ext_matched),
        "item_accuracy":     round(total_credit / gt_count * 100 if gt_count else 0, 1),
        "total_credit":      round(total_credit, 2),
    }


def evaluate_document(extracted: dict, ground_truth: dict,
                      filename: str, proc_time: float) -> tuple[DocumentMetrics, dict]:
    """Compare one extracted JSON against its ground truth."""
    doc = DocumentMetrics(filename=filename, processing_s=proc_time)
    for fpath in EVAL_FIELDS:
        exp_val = _get(ground_truth, fpath)
        if exp_val is _MISSING: continue
        ext_val = _get(extracted, fpath)
        if ext_val is _MISSING: ext_val = None
        is_num  = fpath in NUMERIC_FIELDS
        is_date = fpath in DATE_FIELDS
        outcome = _classify_outcome(ext_val, exp_val, is_num, is_date, fpath)
        doc.field_results.append(FieldResult(field=fpath, extracted=ext_val,
                                             expected=exp_val, outcome=outcome))
    gt_items  = ground_truth.get("line_items", [])
    ext_items = extracted.get("line_items", [])
    item_stats = _compare_line_items(ext_items, gt_items) if gt_items else {}
    return doc, item_stats


def coverage_report(extracted_files: list[Path]) -> dict:
    """Field coverage when no ground truth is available."""
    totals: dict[str, int] = {f: 0 for f in EVAL_FIELDS}
    counts: dict[str, int] = {f: 0 for f in EVAL_FIELDS}
    for fp in extracted_files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for fpath in EVAL_FIELDS:
            totals[fpath] += 1
            v = _get(data, fpath)
            if not _is_empty(v) and v is not _MISSING:
                counts[fpath] += 1
    return {
        f: {
            "extracted_count": counts[f],
            "total_docs":      totals[f],
            "coverage_pct":    round(counts[f] / totals[f] * 100 if totals[f] else 0, 1),
        }
        for f in EVAL_FIELDS
    }


def _count_lines(fp: Path) -> dict:
    try:
        src   = fp.read_text(encoding="utf-8")
        lines = src.splitlines()
        code = blank = comment = 0
        for ln in lines:
            s = ln.strip()
            if not s:               blank   += 1
            elif s.startswith("#"): comment += 1
            else:                   code    += 1
        return {"total_lines": len(lines), "code_lines": code,
                "comment_lines": comment, "blank_lines": blank,
                "comment_ratio": round(comment / max(code, 1) * 100, 1)}
    except Exception:
        return {}


def _count_functions(fp: Path) -> dict:
    try:
        tree    = ast.parse(fp.read_text(encoding="utf-8"))
        funcs   = [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        documented = sum(1 for f in funcs if ast.get_docstring(f))
        return {"functions": len(funcs), "classes": len(classes),
                "documented_funcs": documented,
                "documentation_pct": round(documented / max(len(funcs), 1) * 100, 1)}
    except Exception:
        return {}


def analyze_code_quality(script: Path) -> dict:
    if not script.exists():
        return {"error": f"Script not found: {script}"}
    lines = _count_lines(script)
    funcs = _count_functions(script)
    avg_fn = lines.get("code_lines", 0) / max(funcs.get("functions", 1), 1)
    score  = 100.0
    if lines.get("comment_ratio", 0) < 10:     score -= 10
    if funcs.get("documentation_pct", 0) < 50: score -= 15
    if avg_fn > 50:                             score -= 10
    if lines.get("total_lines", 0) > 1200:     score -= 5
    return {"file": script.name, "line_metrics": lines, "function_metrics": funcs,
            "avg_lines_per_fn": round(avg_fn, 1),
            "maintainability": round(min(score, 100), 1)}


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT RENDERING
# ══════════════════════════════════════════════════════════════════════════════

W = 72

def _hr(c: str = "─") -> str:   return c * W
def _center(t: str) -> str:     return t.center(W)
def _row(label: str, value: Any, width: int = 44) -> str:
    label, value = str(label), str(value)
    return f"  {label}{'.' * max(1, width - len(label))} {value}"
def _bar(pct: float, width: int = 28) -> str:
    pct    = max(0.0, min(100.0, pct))
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def print_full_report(
    doc_metrics:     list[DocumentMetrics],
    item_stats_list: list[dict],
    coverage:        dict | None,
    timing:          dict,
    code_quality:    dict,
    has_gt:          bool,
    gpu_stats:       dict,
) -> dict:
    """Render the full evaluation dashboard and return the report dict."""

    print("\n" + "═" * W)
    print(_center("  HOSPITAL BILL EXTRACTOR — EVALUATION REPORT v5  "))
    print("═" * W)

    overall_acc = precision = recall = f1 = 0.0
    total_tp = total_tn = total_fp = total_fn = 0
    threshold_met = False
    item_acc = 0.0
    agg: dict = {}

    # ── SECTION 1: Accuracy ──────────────────────────────────────────────────
    print(f"\n{'▌ ACCURACY METRICS':^{W}}")
    print(_hr())

    if has_gt and doc_metrics:
        total_tp = sum(d.tp for d in doc_metrics)
        total_tn = sum(d.tn for d in doc_metrics)
        total_fp = sum(d.fp for d in doc_metrics)
        total_fn = sum(d.fn for d in doc_metrics)
        total    = total_tp + total_tn + total_fp + total_fn

        overall_acc = (total_tp + total_tn) / total if total else 0
        precision   = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
        recall      = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
        f1          = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        print(_row("True  Positives  (TP) — Correct extractions",   total_tp))
        print(_row("True  Negatives  (TN) — Correct 'not present'", total_tn))
        print(_row("False Positives  (FP) — Wrong value extracted",  total_fp))
        print(_row("False Negatives  (FN) — Missed real values",     total_fn))
        print(_hr("·"))
        print(_row("Overall Accuracy",     f"{overall_acc*100:.2f}%"))
        print(_row("Precision",            f"{precision*100:.2f}%"))
        print(_row("Recall (Sensitivity)", f"{recall*100:.2f}%"))
        print(_row("F1 Score",             f"{f1*100:.2f}%"))
        print(f"\n  Accuracy  {_bar(overall_acc*100)}  {overall_acc*100:.1f}%")

        threshold_met = overall_acc >= 0.95
        status = "✅ PASS — ≥95% threshold met" if threshold_met else "❌ FAIL — below 95%"
        print(f"\n  Qualification Status: {status}")

        print(f"\n  {'─ Per-Document Breakdown ':─<{W-2}}")
        print(f"  {'File':<32} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4} {'Acc':>7} {'F1':>7}")
        print(f"  {'-'*66}")
        for dm in doc_metrics:
            print(f"  {dm.filename[:31]:<32} {dm.tp:>4} {dm.tn:>4} {dm.fp:>4} {dm.fn:>4}"
                  f" {dm.accuracy*100:>6.1f}% {dm.f1*100:>6.1f}%")

        print(f"\n  {'─ Per-Field Accuracy (⚠ = <80%, fix extractor) ':─<{W-2}}")
        field_stats: dict[str, dict] = {}
        for dm in doc_metrics:
            for fr in dm.field_results:
                s = field_stats.setdefault(fr.field, {"TP":0,"TN":0,"FP":0,"FN":0,"fp_ex":[],"fn_ex":[]})
                s[fr.outcome] += 1
                if fr.outcome == "FP" and len(s["fp_ex"]) < 3:
                    s["fp_ex"].append(f"got={str(fr.extracted)[:30]!r}  exp={str(fr.expected)[:30]!r}")
                if fr.outcome == "FN" and len(s["fn_ex"]) < 3:
                    s["fn_ex"].append(f"exp={str(fr.expected)[:40]!r}")

        print(f"  {'Field':<46} {'TP':>3} {'TN':>3} {'FP':>3} {'FN':>3} {'Acc':>7}")
        print(f"  {'-'*67}")
        for fname, s in sorted(field_stats.items()):
            t   = sum(v for k, v in s.items() if k in ("TP","TN","FP","FN"))
            acc = (s["TP"] + s["TN"]) / t * 100 if t else 0
            flag = " ⚠" if acc < 80 else ""
            print(f"  {fname:<46} {s['TP']:>3} {s['TN']:>3} {s['FP']:>3} {s['FN']:>3} {acc:>6.0f}%{flag}")
            if acc < 80:
                for ex in s["fp_ex"]: print(f"      FP: {ex}")
                for ex in s["fn_ex"]: print(f"      FN: {ex}")

        if item_stats_list:
            print(f"\n  {'─ Line-Item Accuracy (best-of-N fuzzy, Jaccard ≥0.35) ':─<{W-2}}")
            agg = {}
            for k in ["gt_count","extracted_count","matched_full",
                      "matched_desc_only","matched_amt_only","missed","spurious"]:
                agg[k] = sum(d.get(k, 0) for d in item_stats_list)
            agg["total_credit"] = round(sum(d.get("total_credit", 0) for d in item_stats_list), 2)
            item_acc = agg["total_credit"] / agg["gt_count"] * 100 if agg["gt_count"] else 0
            print(_row("GT line items (total)",         agg["gt_count"]))
            print(_row("Extracted line items",          agg["extracted_count"]))
            print(_row("Matched full (desc+amt, ×1.0)", agg["matched_full"]))
            print(_row("Matched desc only (×0.7)",      agg["matched_desc_only"]))
            print(_row("Matched amount only (×0.4)",    agg["matched_amt_only"]))
            print(_row("Missed (FN)",                   agg["missed"]))
            print(_row("Spurious (FP)",                 agg["spurious"]))
            print(f"\n  Item Accuracy (credit-weighted)  {_bar(item_acc)}  {item_acc:.1f}%")

    else:
        print("  ℹ  No ground truth — showing FIELD COVERAGE.")
        if coverage:
            print(f"\n  {'Field':<46} {'Coverage':>10}  Bar")
            print(f"  {'-'*69}")
            for fname, info in coverage.items():
                pct  = info["coverage_pct"]
                flag = " ⚠" if pct < 60 else ""
                print(f"  {fname:<46} {pct:>8.1f}%  {_bar(pct, 16)}{flag}")
            avg_cov = statistics.mean(v["coverage_pct"] for v in coverage.values())
            print(f"\n  Average field coverage: {avg_cov:.1f}%")

    # ── SECTION 2: Response Time ─────────────────────────────────────────────
    print(f"\n\n{'▌ RESPONSE TIME METRICS':^{W}}")
    print(_hr())
    print(_row("Files processed",  timing["total_files"]))
    print(_row("Total wall time",  f"{timing['total_s']:.2f}s"))
    print(_row("Average per file", f"{timing['avg_s']:.2f}s"))
    print(_row("Fastest file",     f"{timing['min_s']:.2f}s"))
    print(_row("Slowest file",     f"{timing['max_s']:.2f}s"))
    if "p50_s" in timing:
        print(_row("Median (P50)",      f"{timing['p50_s']:.2f}s"))
        print(_row("P90 response time", f"{timing['p90_s']:.2f}s"))
    avg = timing["avg_s"]
    speed = ("🚀 Excellent (< 5s/doc)" if avg < 5  else
             "✅ Good (< 15s/doc)"     if avg < 15 else
             "⚠  Acceptable (< 30s)"  if avg < 30 else "❌ Slow (> 30s/doc)")
    print(f"\n  Speed Rating: {speed}")

    # ── SECTION 3: GPU COST (NEW) ────────────────────────────────────────────
    if gpu_stats:
        print_gpu_cost_section(gpu_stats, W)

    # ── SECTION 4: Code Quality ──────────────────────────────────────────────
    print(f"\n\n{'▌ CODE QUALITY METRICS':^{W}}")
    print(_hr())
    if "error" not in code_quality:
        lm = code_quality.get("line_metrics", {})
        fm = code_quality.get("function_metrics", {})
        print(_row("Script",                 code_quality.get("file", "N/A")))
        print(_row("Total lines",            lm.get("total_lines", "N/A")))
        print(_row("Code lines",             lm.get("code_lines", "N/A")))
        print(_row("Comment lines",          lm.get("comment_lines", "N/A")))
        print(_row("Comment ratio",          f"{lm.get('comment_ratio',0):.1f}%"))
        print(_hr("·"))
        print(_row("Functions defined",      fm.get("functions", "N/A")))
        print(_row("Functions with docstrings",
                   f"{fm.get('documented_funcs','N/A')} / {fm.get('functions','?')} "
                   f"({fm.get('documentation_pct',0):.0f}%)"))
        print(_row("Avg lines per function", f"{code_quality.get('avg_lines_per_fn','N/A')}"))
        maint = code_quality.get("maintainability", 0)
        print(f"\n  Maintainability  {_bar(maint)}  {maint:.0f}/100")
        grade = ("A (Excellent)" if maint >= 85 else "B (Good)" if maint >= 70
                 else "C (Fair)" if maint >= 55 else "D (Needs work)")
        print(f"  Grade: {grade}")
    else:
        print(f"  ⚠  {code_quality['error']}")

    # ── SECTION 5: Weighted Scorecard ────────────────────────────────────────
    print(f"\n\n{'▌ SCORECARD SUMMARY (weighted)':^{W}}")
    print(_hr("═"))

    scores:  dict[str, float] = {}
    weights: dict[str, float] = {}

    if has_gt and doc_metrics:
        scores["Accuracy (field)"]   = overall_acc * 100;  weights["Accuracy (field)"]   = 2.0
        if item_stats_list:
            scores["Line-Item Accuracy"] = item_acc;        weights["Line-Item Accuracy"] = 1.5
    else:
        avg_cov = statistics.mean(v["coverage_pct"] for v in coverage.values()) if coverage else 0
        scores["Field Coverage (proxy)"] = avg_cov;         weights["Field Coverage (proxy)"] = 2.0

    scores["Cost Efficiency"]  = 100.0;              weights["Cost Efficiency"]  = 1.0
    scores["Response Time"]    = min(100, max(0, 100 - (timing["avg_s"] / 30) * 50))
    weights["Response Time"]   = 1.0
    scores["Code Quality"]     = code_quality.get("maintainability", 70)
    weights["Code Quality"]    = 1.0

    weighted_sum  = sum(scores[k] * weights[k] for k in scores)
    total_weight  = sum(weights.values())
    overall_score = weighted_sum / total_weight

    for metric, score in scores.items():
        w     = weights[metric]
        label = f"{metric} (×{w:.0f})" if w != 1.0 else metric
        print(f"  {label:<36}  {_bar(score, 24)}  {score:.1f}/100")
    print(_hr("·"))
    print(f"  {'WEIGHTED OVERALL SCORE':<36}  {_bar(overall_score, 24)}  {overall_score:.1f}/100")
    print("═" * W + "\n")

    # Build report dict
    report: dict[str, Any] = {
        "generated_at":  __import__("datetime").datetime.now().isoformat(),
        "timing":        timing,
        "gpu_cost":      gpu_stats,
        "cost":          {"total_usd": 0.0, "per_doc": 0.0, "engine": "tesseract_local"},
        "code_quality":  code_quality,
        "scores":        {k: round(v, 2) for k, v in scores.items()},
        "weights":       weights,
        "overall_score": round(overall_score, 2),
    }

    if has_gt and doc_metrics:
        report["accuracy"] = {
            "TP": total_tp, "TN": total_tn, "FP": total_fp, "FN": total_fn,
            "overall_pct":   round(overall_acc * 100, 2),
            "precision_pct": round(precision   * 100, 2),
            "recall_pct":    round(recall      * 100, 2),
            "f1_pct":        round(f1          * 100, 2),
            "threshold_met": threshold_met,
        }
        report["per_document"] = [
            {"file": dm.filename, "time_s": dm.processing_s,
             "TP": dm.tp, "TN": dm.tn, "FP": dm.fp, "FN": dm.fn,
             "accuracy": round(dm.accuracy * 100, 2), "f1": round(dm.f1 * 100, 2)}
            for dm in doc_metrics
        ]
        if item_stats_list:
            report["line_item_accuracy"] = {**agg, "pct": round(item_acc, 2)}
    else:
        report["field_coverage"] = coverage

    return report


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point: load files, evaluate, print dashboard, save report."""
    print("=" * W)
    print(_center("  EVALUATION ENGINE v5 STARTING  "))
    print("=" * W)

    extracted_files = sorted([
        f for f in OUT_DIR.glob("*.json") if not f.name.startswith("_")
    ])
    if not extracted_files:
        print(f"\n  ✗  No extracted JSON files found in {OUT_DIR}")
        sys.exit(1)
    print(f"\n  Found {len(extracted_files)} extracted file(s) in {OUT_DIR}")

    # ── Collect timing AND gpu cost from _meta ────────────────────────────
    times: list[float]     = []
    gpu_costs_inr: list[float] = []

    for fp in extracted_files:
        try:
            meta = json.loads(fp.read_text(encoding="utf-8")).get("_meta", {})
            t = meta.get("processing_time_s")
            if t is not None:
                times.append(float(t))
            # gpu_cost_inr present in v11+; absent in older versions
            c = meta.get("gpu_cost_inr")
            if c is not None:
                gpu_costs_inr.append(float(c))
        except Exception:
            pass

    if not times:
        times = [0.0]

    timing: dict[str, Any] = {
        "total_files": len(extracted_files),
        "total_s":     round(sum(times), 2),
        "avg_s":       round(statistics.mean(times), 2),
        "min_s":       round(min(times), 2),
        "max_s":       round(max(times), 2),
    }
    if len(times) >= 2:
        st = sorted(times)
        timing["p50_s"] = round(statistics.median(st), 2)
        timing["p90_s"] = round(st[math.ceil(len(st) * 0.9) - 1], 2)

    # Build GPU stats — use real measured costs where available
    total_gpu_seconds = sum(times)
    total_gpu_cost    = sum(gpu_costs_inr)   # 0.0 if running on v10 JSONs
    gpu_stats = _build_gpu_summary(total_gpu_seconds, len(extracted_files), total_gpu_cost)

    # ── Ground truth evaluation ───────────────────────────────────────────
    gt_files = list(GROUND_TRUTH.glob("*.json")) if GROUND_TRUTH.exists() else []
    has_gt   = len(gt_files) > 0

    doc_metrics:     list[DocumentMetrics] = []
    item_stats_list: list[dict]            = []
    coverage:        dict | None           = None

    if has_gt:
        print(f"  Found {len(gt_files)} ground-truth file(s) — full accuracy mode.\n")
        for gt_fp in gt_files:
            ext_fp = OUT_DIR / gt_fp.name
            if not ext_fp.exists():
                print(f"  ⚠  No extracted file for GT: {gt_fp.name} — skipping")
                continue
            try:
                extracted = json.loads(ext_fp.read_text(encoding="utf-8"))
                gt        = json.loads(gt_fp.read_text(encoding="utf-8"))
                proc_time = extracted.get("_meta", {}).get("processing_time_s", 0)
                dm, istats = evaluate_document(extracted, gt, gt_fp.name, proc_time)
                doc_metrics.append(dm)
                if istats: item_stats_list.append(istats)
            except Exception as exc:
                import traceback as tb
                print(f"  ✗  Error evaluating {gt_fp.name}: {exc}")
                print(tb.format_exc())
    else:
        print("  ℹ  No ground truth — computing field coverage.\n")
        coverage = coverage_report(extracted_files)

    cq     = analyze_code_quality(EXTRACTOR_SCRIPT)
    report = print_full_report(
        doc_metrics, item_stats_list, coverage,
        timing, cq, has_gt, gpu_stats,
    )

    report_path = OUT_DIR / "_evaluation_report_v5.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Full report saved → {report_path}\n")


if __name__ == "__main__":
    main()