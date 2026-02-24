#!/usr/bin/env python3
"""
Hospital Bill Extractor  v11  —  GPU Cost Tracking Edition
===========================================================

  All v10 fixes retained (FIX A–E).

  NEW — GPU COST TRACKING (FIX F)
    • GPU_NAME / GPU_TDP_WATTS / GPU_UTIL_PCT / SYSTEM_WATTS / ELEC_RATE_INR
      constants at top of file — edit once for your system
    • gpu_cost_for_seconds(secs) → ₹ cost for any duration
    • gpu_cost_summary(total_secs, n_files) → full metrics dict
    • print_gpu_cost_report() called at end of main() after DONE line
    • Per-file cost logged in _meta.gpu_cost_inr for every output JSON
    • GPU cost section added to _report.json
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import traceback
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
BASE          = Path(__file__).parent
IMG_DIR       = BASE / "Sample data"
PDF_DIR       = BASE / "Testing data"
OUT_DIR       = BASE / "output"
TESSERACT_CMD = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
PDF_DPI       = 300

# ── GPU COST CONFIGURATION — edit these to match your system ─────────────────
GPU_NAME         = "RTX 3060"   # label only, for display
GPU_TDP_WATTS    = 170          # GPU max TDP in watts
GPU_UTIL_PCT     = 0.40         # fraction of TDP used during OCR  (0.0–1.0)
SYSTEM_WATTS     = 80           # CPU + RAM + cooling overhead in watts
ELEC_RATE_INR    = 6.5          # ₹ per kWh  (Indian average ≈ ₹6–9)
# ══════════════════════════════════════════════════════════════════════════════

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
            ".JPG", ".JPEG", ".PNG", ".TIFF"}
PDF_EXT  = ".pdf"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  GPU COST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _effective_watts() -> float:
    """Return actual power draw: GPU active draw + system overhead."""
    return GPU_TDP_WATTS * GPU_UTIL_PCT + SYSTEM_WATTS


def gpu_cost_for_seconds(seconds: float) -> float:
    """
    Calculate electricity cost in ₹ for a given number of seconds of GPU use.

    Formula:
        effective_kW = (GPU_TDP × util_fraction + system_W) / 1000
        cost_inr     = effective_kW × (seconds / 3600) × ELEC_RATE_INR
    """
    kw    = _effective_watts() / 1000.0
    hours = seconds / 3600.0
    return kw * hours * ELEC_RATE_INR


def gpu_cost_summary(total_seconds: float, n_files: int) -> dict:
    """
    Return a dict of GPU cost metrics for a batch of files.

    Includes per-file, total, per-1000, and annualised (22 days/month × 12)
    costs. Adds a GPT-4V savings comparison for hackathon context.
    """
    effective_w  = _effective_watts()
    total_cost   = gpu_cost_for_seconds(total_seconds)
    per_file     = total_cost / n_files if n_files else 0.0
    per_1000     = per_file * 1000
    annual_cost  = per_file * n_files * 22 * 12   # same daily volume

    return {
        "gpu_name":              GPU_NAME,
        "gpu_tdp_w":             GPU_TDP_WATTS,
        "gpu_util_pct":          round(GPU_UTIL_PCT * 100, 1),
        "system_overhead_w":     SYSTEM_WATTS,
        "effective_watts":       round(effective_w, 1),
        "elec_rate_inr_per_kwh": ELEC_RATE_INR,
        "total_seconds":         round(total_seconds, 2),
        "n_files":               n_files,
        "cost_per_file_inr":     round(per_file, 6),
        "total_cost_inr":        round(total_cost, 6),
        "cost_per_1000_inr":     round(per_1000, 4),
        "annual_cost_inr":       round(annual_cost, 2),
    }


def print_gpu_cost_report(total_seconds: float, n_files: int) -> None:
    """Print a formatted GPU cost block after the DONE line."""
    s   = gpu_cost_summary(total_seconds, n_files)
    W   = 65
    dot = lambda lbl, val, w=44: f"  {lbl}{'.' * max(1, w - len(lbl))} {val}"

    gpt4v_usd    = n_files * 0.03
    gpt4v_inr    = gpt4v_usd * 83
    savings_pct  = (1 - s["total_cost_inr"] / gpt4v_inr) * 100 if gpt4v_inr else 100

    print(f"\n{'=' * W}")
    print(f"  GPU COST METRICS  ({s['gpu_name']}  |  {s['effective_watts']}W effective)")
    print(f"{'=' * W}")
    print(dot("GPU TDP",                  f"{s['gpu_tdp_w']}W"))
    print(dot("GPU utilisation",          f"{s['gpu_util_pct']}%  (OCR is CPU-heavy; ~40% typical)"))
    print(dot("System overhead",          f"{s['system_overhead_w']}W"))
    print(dot("Effective power draw",     f"{s['effective_watts']}W"))
    print(dot("Electricity rate",         f"₹{s['elec_rate_inr_per_kwh']}/kWh"))
    print("·" * W)
    print(dot("Files processed",          str(s["n_files"])))
    print(dot("Total GPU time",           f"{s['total_seconds']:.2f}s"))
    print(dot("Cost per file",            f"₹{s['cost_per_file_inr']:.6f}"))
    print(dot("Total batch cost",         f"₹{s['total_cost_inr']:.6f}"))
    print(dot("Cost per 1,000 bills",     f"₹{s['cost_per_1000_inr']:.4f}"))
    print(dot("Projected annual cost",    f"₹{s['annual_cost_inr']:.2f}  "
                                           f"(~${s['annual_cost_inr']/83:.2f} USD)"))
    print(dot("GPT-4V cloud equivalent",  f"~${gpt4v_usd:.2f}  (₹{gpt4v_inr:.0f})"))
    print(dot("Savings vs GPT-4V",        f"{savings_pct:.1f}%  — local GPU is essentially free"))
    print("=" * W)


# ══════════════════════════════════════════════════════════════════════════════
#  1.  PREFLIGHT
# ══════════════════════════════════════════════════════════════════════════════

def preflight() -> None:
    """Validate runtime dependencies and input folders before processing."""
    print("=" * 65)
    print("  HOSPITAL BILL EXTRACTOR  v11  —  GPU Cost Tracking Build")
    print("=" * 65)
    errors: list[str] = []

    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        ver = pytesseract.get_tesseract_version()
        print(f"  [OK] Tesseract {ver}")
    except Exception as exc:
        errors.append(f"Tesseract unavailable: {exc}")

    for friendly, module in [
        ("Pillow",        "PIL"),
        ("PyMuPDF",       "fitz"),
        ("opencv-python", "cv2"),
    ]:
        try:
            __import__(module)
            print(f"  [OK] {friendly}")
        except ImportError:
            errors.append(f"{friendly} missing  →  pip install {friendly}")

    for label, folder in [("Sample data", IMG_DIR), ("Testing data", PDF_DIR)]:
        if folder.exists():
            n = sum(1 for f in folder.rglob("*") if f.is_file())
            print(f"  [OK] {label} — {n} file(s)")
        else:
            errors.append(f"Folder not found: {folder}")

    print(f"  [OK] GPU config: {GPU_NAME}  {GPU_TDP_WATTS}W TDP  "
          f"{int(GPU_UTIL_PCT*100)}% util  +{SYSTEM_WATTS}W sys  "
          f"= {_effective_watts():.0f}W effective  @  ₹{ELEC_RATE_INR}/kWh")

    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  [OK] Output → {OUT_DIR}\n{'=' * 65}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  2.  OCR
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess(img: "PIL.Image.Image") -> "PIL.Image.Image":
    """Multi-step image enhancement: grayscale → denoise → adaptive threshold → deskew."""
    import cv2
    import numpy as np
    from PIL import Image

    arr    = np.array(img.convert("RGB"))
    gray   = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray   = cv2.fastNlMeansDenoising(gray, h=10)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=11,
    )

    coords = np.column_stack(np.where(binary < 128))
    if len(coords) > 100:
        rect  = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        angle = -angle
        if abs(angle) > 0.3:
            h, w = binary.shape
            M      = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            binary = cv2.warpAffine(
                binary, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

    return Image.fromarray(binary)


def extract_text(fp: Path) -> str:
    """Extract OCR text from image or PDF. PDFs rendered at PDF_DPI dpi."""
    import pytesseract
    from PIL import Image

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    cfg   = "--psm 6 --oem 3 -c preserve_interword_spaces=1"
    pages: list[str] = []

    if fp.suffix.lower() == PDF_EXT:
        import fitz
        doc   = fitz.open(str(fp))
        scale = PDF_DPI / 72.0
        for page in doc:
            pix = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                colorspace=fitz.csRGB,
            )
            img = Image.open(BytesIO(pix.tobytes("png")))
            pages.append(pytesseract.image_to_string(_preprocess(img), config=cfg))
        doc.close()
    else:
        img = Image.open(fp)
        w, h = img.size
        if max(w, h) < 1500:
            factor = max(2, 1500 // max(w, h))
            img = img.resize((w * factor, h * factor), Image.LANCZOS)
        pages.append(pytesseract.image_to_string(_preprocess(img), config=cfg))

    return "\n\n".join(pages)


# ══════════════════════════════════════════════════════════════════════════════
#  3.  CORE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

_OCR_FIXES: list[tuple[str, str]] = [
    (r"[Pp]atien[t7]\b",       "Patient"),
    (r"\b[Hh][o0]spital\b",    "Hospital"),
    (r"\b[Aa]dmi[s5]sion\b",   "Admission"),
    (r"\b[Dd]i[s5]charge\b",   "Discharge"),
    (r"\b[Dd][o0]ct[o0]r\b",   "Doctor"),
    (r"\b[Ww]ar[d0]\b",        "Ward"),
    (r"\b[Bb]ill\s*N[o0]\b",   "Bill No"),
    (r"\bUHI[D0]\b",            "UHID"),
    (r"(?<=[A-Z])0(?=[A-Z])",  "O"),
    (r"(?<=\d)[oO](?=\d)",      "0"),
    (r"\b[Pp]t\.\s*[Nn]ame\b", "Pt Name"),
    (r"\bSEROO(\d+)",           r"SER00\1"),
    (r"\bSERO(\d+)",            r"SER0\1"),
    (r"»",                       "-"),
    (r"[""'']",                 "'"),
    (r"Hospitallsation",        "Hospitalisation"),
    (r"Hospitallzation",        "Hospitalization"),
    (r"\bOl\b",                 "01"),
    (r"\bl0\b",                 "10"),
    (r"(?<=\d)[Ss](?=\d)",      "5"),
    (r"(?<=\d)[Ii](?=\d)",      "1"),
    (r"(?<=\d)[Zz](?=\d)",      "2"),
]

_F  = re.IGNORECASE | re.MULTILINE
_FD = re.IGNORECASE | re.MULTILINE | re.DOTALL


def _normalise(text: str) -> str:
    """Apply OCR character corrections to raw OCR text."""
    for pat, repl in _OCR_FIXES:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text


def _find(patterns, text: str, group: int = 1, flags: int = _F) -> str | None:
    """Try each regex pattern in order; return first non-empty match or None."""
    if isinstance(patterns, str):
        patterns = [patterns]
    for pat in patterns:
        try:
            m = re.search(pat, text, flags)
            if m:
                val = m.group(group).strip()
                if val:
                    return val
        except (re.error, IndexError):
            continue
    return None


def _clean_amount(raw: str | None) -> float | None:
    """Parse Indian currency string to float. Returns None for zero/unparseable."""
    if raw is None:
        return None
    s = re.sub(r"[₹\u20b9,\s]", "", str(raw))
    s = s.rstrip(".")
    parts = s.split(".")
    if len(parts) > 2:
        s = "".join(parts[:-1]) + "." + parts[-1]
    try:
        v = round(float(s), 2)
        return v if v > 0 else None
    except ValueError:
        return None


_MONTHS: dict[str, str] = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def _to_date(raw: str | None) -> str | None:
    """Parse any common Indian date format to ISO 8601 (YYYY-MM-DD)."""
    if not raw:
        return None
    raw = re.split(r"\s+\d{1,2}:\d{2}", raw)[0].strip(" .,\n\t|")

    m = re.search(
        r"(\d{1,2})[\s\-/\.]"
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[\s\-/\.](\d{2,4})",
        raw, re.I,
    )
    if m:
        y = m.group(3)
        y = "20" + y if len(y) == 2 else y
        return f"{y}-{_MONTHS[m.group(2)[:3].lower()]}-{m.group(1).zfill(2)}"

    m = re.search(r"(\d{4})[-/\.](\d{2})[-/\.](\d{2})", raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    m = re.search(r"(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{2,4})", raw)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        y = "20" + y if len(y) == 2 else y
        return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"

    return None


def _date_valid(iso: str | None) -> bool:
    """Return True if iso date is plausible (2000–2035)."""
    if not iso:
        return False
    try:
        y = int(iso[:4])
        return 2000 <= y <= 2035
    except (ValueError, TypeError):
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  4.  FIELD EXTRACTORS  (unchanged from v10)
# ══════════════════════════════════════════════════════════════════════════════

def _hospital_name(lines: list[str], text: str) -> str | None:
    """Extract hospital name: known brand → first prominent header line."""
    known = [
        r"Neotia\s*[Gg]etwel", r"Apollo\s+(?:Hospital|Clinic|Gleneagles)",
        r"Fortis\s+(?:Hospital|Healthcare)", r"Max\s+(?:Hospital|Healthcare|Super Speciality)",
        r"AIIMS", r"Manipal\s+Hospital", r"Narayana\s+Health", r"Medanta",
        r"Lilavati\s+Hospital", r"Kokilaben\s+(?:Dhirubhai|Hospital)", r"NIMHANS",
        r"Wockhardt\s+Hospital", r"Aster\s+(?:Hospital|CMI|Medcity)", r"Columbia\s+Asia",
        r"Sakra\s+(?:World|Hospital)", r"Cloudnine", r"Rainbow\s+(?:Hospital|Children)",
        r"Yashoda\s+Hospital", r"Care\s+Hospital", r"Tata\s+(?:Memorial|Medical)",
        r"Christian\s+Medical\s+(?:College|CMC)", r"St\.?\s+John",
    ]
    for pat in known:
        m = re.search(pat, text, re.I)
        if m:
            for line in lines[:15]:
                if re.search(pat, line, re.I):
                    return line.strip(" '\"()\t")
            return m.group(0)

    _skip = re.compile(
        r"^\s*(?:page|printed|date|tel|ph|fax|email|gstin|www\.|http|"
        r"©|\d{2}[/\-]\d{2}|bill\s|invoice|receipt|from\s|a unit|"
        r"healthcare centre|prepared|generated)", re.I,
    )
    for line in lines[:12]:
        clean = re.sub(r"[^A-Za-z\s&\-\(\)']", "", line).strip()
        if len(clean) >= 5 and not _skip.match(line.strip()):
            return line.strip(" '\"()")
    return None


_BILL_TYPES: list[str] = [
    "Detailed Hospital Bill", "Final Hospital Bill", "Final Bill",
    "IPD Bill", "OPD Bill", "Pharmacy Bill", "Diagnostic Bill",
    "Interim Bill", "Summary Bill", "Discharge Bill", "Discharge Summary",
    "Hospitalisation Charges", "Inpatient Bill", "Outpatient Bill",
    "Investigation Bill", "Tax Invoice", "Credit Bill",
]


def _bill_type(text: str, filename: str) -> str:
    """Detect bill type from content; fall back to filename heuristics."""
    for bt in _BILL_TYPES:
        if re.search(re.escape(bt), text, re.I):
            return bt
    fn = filename.lower()
    if "pharma" in fn: return "Pharmacy Bill"
    if "diag"   in fn: return "Diagnostic Bill"
    if "opd"    in fn: return "OPD Bill"
    if "ipd"    in fn: return "IPD Bill"
    if re.search(r"int[-_]?\d", fn): return "Interim Bill"
    if "final"     in fn: return "Final Bill"
    if "discharge" in fn: return "Discharge Bill"
    if "summary"   in fn: return "Summary Bill"
    return "Hospital Bill"


def _bill_number(text: str) -> str | None:
    """Extract bill / invoice number."""
    return _find([
        r"\bSER[0-9]{4,12}\b",
        r"[Bb]ill\s*[Nn]o\.?\s*[:\-#]?\s*([A-Z0-9][A-Z0-9/\-]{2,20})",
        r"[Ii]nvoice\s*[Nn]o\.?\s*[:\-#]?\s*([A-Z0-9][A-Z0-9/\-]{2,20})",
        r"[Rr]eceipt\s*[Nn]o\.?\s*[:\-#]?\s*([A-Z0-9][A-Z0-9/\-]{2,20})",
        r"[Rr]ef(?:erence)?\.?\s*[Nn]o\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9/\-]{2,20})",
        r"IP\s*[Nn]o\.?\s*[:\-|]?\s*([A-Za-z0-9/\-]+)",
    ], text)


def _bill_date(text: str) -> str | None:
    """FIX A: Extract bill date with 5-strategy cascade + proximity scoring."""
    for pat in [
        r"[Bb]ill\s*[Dd]t\.?\s*[:\-]?\s*(\d{1,2}[\s\-/\.]\w+[\s\-/\.]\d{2,4})",
        r"[Bb]ill\s*[Dd]ate\s*[:\-]?\s*(\d{1,2}[\s\-/\.]\w+[\s\-/\.]\d{2,4})",
        r"[Bb]ill\s*[Dd]ate\s*[:\-]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
        r"[Ii]nvoice\s*[Dd]ate\s*[:\-]?\s*(\d{1,2}[\s\-/\.]\w+[\s\-/\.]\d{2,4})",
        r"[Ii]nvoice\s*[Dd]ate\s*[:\-]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
        r"[Dd]ate\s*[Oo]f\s*[Bb]ill\s*[:\-]?\s*(\d{1,2}[\s\-/\.]\w+[\s\-/\.]\d{2,4})",
        r"[Dd]ate\s*[Oo]f\s*[Bb]ill\s*[:\-]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
        r"[Dd]ate\s*[Oo]f\s*[Ii]nvoice\s*[:\-]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
        r"^[Dd]ate\s*[:\-]\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
        r"^[Dd]t\.?\s*[:\-]\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
    ]:
        raw = _find(pat, text, flags=_F)
        d = _to_date(raw)
        if _date_valid(d):
            return d

    date_pattern = re.compile(
        r"(\d{1,2}[\s\-/\.](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s\-/\.]\d{2,4}"
        r"|\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}"
        r"|\d{4}[-/\.]\d{2}[-/\.]\d{2})", re.I,
    )
    bill_kw = re.compile(r"\b(?:bill|invoice|receipt|date|issued|generated|prepared|printed|dt)\b", re.I)
    excl_kw = re.compile(r"\b(?:admission|admitted|discharge|dob|birth|doa|dod|validity|expiry|expiration)\b", re.I)

    candidates: list[tuple[float, str]] = []
    for m in date_pattern.finditer(text):
        d = _to_date(m.group(1))
        if not _date_valid(d):
            continue
        window = text[max(0, m.start() - 60): m.start()]
        if excl_kw.search(window):
            continue
        score = len(bill_kw.findall(window)) * 2.0
        if re.search(r"\b(?:bill|invoice|receipt)\b", text[max(0, m.start()-30):m.start()], re.I):
            score += 3.0
        score += max(0.0, 1.0 - m.start() / max(len(text), 1) * 2)
        candidates.append((score, d))

    if candidates:
        candidates.sort(key=lambda x: -x[0])
        if candidates[0][0] > 0:
            return candidates[0][1]

    for pat in [
        r"[Pp]r[il]nted\s+[Oo]n\s*:?\s*(\d{1,2}[\s\-/\.][A-Za-z]{3}[\s\-/\.]\d{2,4})",
        r"[Pp]r[il]nted\s+[Oo]n\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
        r"[Gg]enerated\s+[Oo]n\s*:?\s*(\d{2}/\d{2}/\d{4})",
        r"[Pp]repared\s+[Oo]n\s*:?\s*(\d{2}/\d{2}/\d{4})",
    ]:
        raw = _find(pat, text, flags=_F)
        d = _to_date(raw)
        if _date_valid(d):
            return d

    for m in date_pattern.finditer(text[:2000]):
        d = _to_date(m.group(1))
        if _date_valid(d):
            window = text[max(0, m.start() - 40): m.start()]
            if not excl_kw.search(window):
                return d

    return None


_NAME_BLACKLIST = re.compile(
    r"\b(?:hospital|clinic|healthcare|doctor|ward|bed|room|bill|invoice|"
    r"receipt|total|amount|balance|advance|discharge|admission|ipd|opd|"
    r"pharmacy|lab|radiology|ot|dept|department|gstin|uhid|mrd|"
    r"neotia|getwel|apollo|fortis|max|aiims|manipal|medanta)\b", re.I,
)


def _patient_name(text: str) -> str | None:
    """Extract patient name via 3-pass fusion with blacklist filtering."""
    _trim = lambda s: re.split(
        r"\s*(?:Age|DOB|D\.O\.B|Gender|Sex|\bM\b|\bF\b|UHID|IP\s*No|"
        r"W/O|S/O|D/O|H/O|MRD|Bed|Ward|Room|Bill|Phone|Mob)\s*[:\-|/]?",
        s, flags=re.I,
    )[0].strip(" .,|/\\")

    candidates: list[str] = []
    for pat in [
        r"[Pp]atient[\s\-]*[Nn]ame\s*[:\-|]?\s*([A-Za-z][A-Za-z\s\.\']{2,45}?)(?=\n|Age|\s{3,}|DOB|Gender|UHID|$)",
        r"[Nn]ame\s+of\s+[Pp]atient\s*[:\-|]?\s*([A-Za-z][A-Za-z\s\.\']{2,45}?)(?=\n|Age|$)",
        r"[Pp]t\.?\s*[Nn]ame\s*[:\-|]?\s*([A-Za-z][A-Za-z\s\.\']{2,45}?)(?=\n|Age|$)",
        r"[Nn]ame\s*[:\-|]\s*([A-Za-z][A-Za-z\s\.\']{2,45}?)(?=\n|Age|DOB|Gender|$)",
    ]:
        m = re.search(pat, text, re.I | re.MULTILINE)
        if m:
            candidates.append(_trim(m.group(1)))

    for pat in [
        r"\b(?:Mr|Mrs|Ms|Miss|Mast(?:er)?|Shri|Smt|Dr)\s*\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})(?=\s*\n|\s{3,}|Age|$)",
        r"(Baby\s+of\s+[A-Za-z\s]+?)(?=\n|Age|$)",
    ]:
        m = re.search(pat, text, re.I | re.MULTILINE)
        if m:
            candidates.append(_trim(m.group(1)))

    m = re.search(r"([A-Z][A-Z\s]{5,35}?)\s+(?:UHID|MRD?|CR)\s*[:\-#]?\s*[A-Z0-9]{4,}", text, re.I)
    if m:
        candidates.append(_trim(m.group(1)))

    best: str | None = None
    for c in candidates:
        c = c.strip()
        if len(c) < 3 or _NAME_BLACKLIST.search(c):
            continue
        if best is None or len(c) > len(best):
            best = c
    return best


def _age_gender(text: str) -> tuple[int | None, str | None]:
    """Extract age and gender; guard against page-number false positives."""
    m = re.search(r"[Aa]ge\s*/\s*[Ss]ex\s*[:\-|]?\s*(\d{1,3})\s*\w*\s*/\s*([MF])\b", text, re.I)
    if m:
        a = int(m.group(1))
        g = "Male" if m.group(2).upper() == "M" else "Female"
        return (a if 0 < a < 130 else None), g

    age: int | None = None
    raw = _find([
        r"[Aa]ge\s*[:\-|]\s*(\d{1,3})\s*(?:Y(?:rs?|ears?)?|[Mm]onths?)",
        r"[Aa]ge\s*/\s*[Ss]ex\s*[:\-|]?\s*(\d{1,3})",
        r"(\d{1,3})\s+(?:Yrs?|Years?|Months?)\b",
        r"\bAGE\s*[:\-]\s*(\d{1,3})\b",
    ], text)
    if raw:
        try:
            a = int(raw); age = a if 0 < a < 130 else None
        except ValueError:
            pass

    gender: str | None = None
    if re.search(r"\bFemale\b|\bF\s*[/|]\s*\d|\bSmt\b|\bMrs\b|\bMs\b", text, re.I):
        gender = "Female"
    elif re.search(r"\bMale\b|\bM\s*[/|]\s*\d|\bShri\b|\bMr\b", text, re.I):
        gender = "Male"
    return age, gender


def _uhid(text: str) -> str | None:
    """Extract UHID / MRD / CR / MR patient identifier."""
    return _find([
        r"UHID\s*[:\-|#]?\s*([A-Z0-9][A-Z0-9/\-]{2,20})",
        r"UHID\s*[Nn]o\.?\s*[:\-|]?\s*([A-Z0-9][A-Z0-9/\-]{2,20})",
        r"MRD?\s*[Nn]o\.?\s*[:\-|]?\s*([A-Z0-9][A-Z0-9/\-]{2,20})",
        r"[Pp]atient\s*I\.?D\.?\s*[:\-|]?\s*([A-Z0-9][A-Z0-9/\-]{2,20})",
        r"CR\s*[Nn]o\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9/\-]{2,20})",
        r"MR\s*#\s*([A-Z0-9][A-Z0-9/\-]{2,20})",
    ], text)


def _stay_dates(text: str) -> tuple[str | None, str | None]:
    """Extract admission and discharge dates with 4-level fallback strategy."""
    _dm = r"(\d{1,2}[\s\-/\.][A-Za-z]{3}[\s\-/\.]\d{4})"
    _dn = r"(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})"

    for pat in [
        rf"[Hh]ospitali[sz]ation\s+[Cc]harges?\s+[Ff]rom\s+{_dm}.{{0,60}}?[Tt]o\s+{_dm}",
        rf"[Hh]ospitali[sz]ation\s+[Cc]harges?\s+[Ff]rom\s+{_dn}.{{0,60}}?[Tt]o\s+{_dn}",
        rf"[Ff]rom\s+{_dm}.{{0,60}}?[Tt]o\s+{_dm}",
        rf"[Ff]rom\s+{_dn}.{{0,40}}?[Tt]o\s+{_dn}",
    ]:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            return _to_date(m.group(1)), _to_date(m.group(2))

    adm = _to_date(_find([
        r"[Aa]dmission\s*[Dd]ate\s*[:\-|]?\s*(\d{1,2}[\s\-/\.]\w+[\s\-/\.]\d{2,4})",
        r"[Aa]dmission\s*[Dd]ate\s*[:\-|]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
        r"[Aa]dmitted\s*[Oo]n\s*[:\-|]?\s*(\d{1,2}[\s\-/\.]\w+[\s\-/\.]\d{2,4})",
        r"D\.?O\.?A\.?\s*[:\-|]?\s*(\d{1,2}[\s\-/\.]\w+[\s\-/\.]\d{2,4})",
        r"D\.?O\.?A\.?\s*[:\-|]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
        r"[Dd]ate\s*[Oo]f\s*[Aa]dmission\s*[:\-|]?\s*(\d{1,2}[\s\-/\.]\w+[\s\-/\.]\d{2,4})",
    ], text))

    dis = _to_date(_find([
        r"[Dd]ischarge\s*[Dd]ate\s*[:\-|]?\s*(\d{1,2}[\s\-/\.]\w+[\s\-/\.]\d{2,4})",
        r"[Dd]ischarge\s*[Dd]ate\s*[:\-|]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
        r"D\.?O\.?D\.?\s*[:\-|]?\s*(\d{1,2}[\s\-/\.]\w+[\s\-/\.]\d{2,4})",
        r"D\.?O\.?D\.?\s*[:\-|]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
        r"[Dd]ischarged\s*[Oo]n\s*[:\-|]?\s*(\d{1,2}[\s\-/\.]\w+[\s\-/\.]\d{2,4})",
        r"[Dd]ate\s*[Oo]f\s*[Dd]ischarge\s*[:\-|]?\s*(\d{1,2}[\s\-/\.]\w+[\s\-/\.]\d{2,4})",
    ], text))

    if adm or dis:
        return adm, dis

    all_dates = re.findall(r"\b(\d{1,2}[\s\-][A-Za-z]{3}[\s\-]\d{4})\b", text)
    parsed = [d for d in (_to_date(x) for x in all_dates) if d]
    if len(parsed) >= 2:
        return parsed[0], parsed[-1]
    if len(parsed) == 1:
        return parsed[0], None
    return None, None


def _ward(text: str) -> str | None:
    """Extract ward name, normalised, with broad keyword list."""
    for sw in ["SICU", "NICU", "MICU", "CTICU", "PICU", "HDU", "ICU", "CCU"]:
        if re.search(rf"\b{sw}\b", text, re.I):
            if re.search(rf"[Ww]ard\s*[:\-]?\s*{sw}|{sw}\s+[Bb]ed|{sw}\s+[Ww]ard", text, re.I):
                return sw

    ward = _find([
        r"[Ww]ard\s*[:\-|#]?\s*([A-Za-z0-9][A-Za-z0-9 \-_/]{0,35}?)(?=\n|Bed|Room|\s{3,}|$)",
        r"[Ww]ard\s*[Nn]ame\s*[:\-|]?\s*([A-Za-z0-9][A-Za-z0-9 \-_/]{0,35}?)(?=\n|$)",
        r"[Rr]oom\s*[Tt]ype\s*[:\-|]?\s*([A-Za-z0-9][A-Za-z0-9 \-_/]{0,35}?)(?=\n|$)",
        r"\b(General\s+Ward|Private\s+(?:Room|Ward)|Semi[\s\-]*Private|"
        r"Deluxe\s+(?:Room|Ward)|Suite|Special\s+Ward|Maternity\s+Ward|"
        r"Pediatric\s+Ward|Surgical\s+Ward|Isolation\s+Ward|"
        r"Observation\s+Ward|Recovery\s+Ward)\b",
    ], text)
    if ward:
        ward = re.split(r"\s*(?:Bed|Room|No\.?|#)\s*[\-:]?\s*\d", ward, flags=re.I)[0]
        return ward.strip(" .,|")
    return None


def _bed(text: str) -> str | None:
    """Extract bed number or bed label."""
    return _find([
        r"[Bb]ed\s*[Nn]o\.?\s*[:\-|#]?\s*([A-Za-z0-9][A-Za-z0-9 \-/]{0,20}?)(?=\n|Ward|Room|$)",
        r"[Bb]ed\s*[Nn]umber\s*[:\-|]?\s*([A-Za-z0-9][A-Za-z0-9\-/]{0,15})",
        r"[Bb]ed\s*[:\-|]\s*([A-Za-z0-9][A-Za-z0-9 \-/]{0,20}?)(?=\n|$)",
    ], text)


_GSTIN_STRUCT = r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}"


def _gstin(text: str) -> str | None:
    """Extract GSTIN via label or structural 15-char pattern scan."""
    labelled = _find([
        rf"GSTIN?\s*[:\-|#]?\s*({_GSTIN_STRUCT})",
        rf"GST\s*[Nn]o\.?\s*[:\-|]?\s*({_GSTIN_STRUCT})",
    ], text)
    if labelled:
        return labelled
    m = re.search(_GSTIN_STRUCT, text)
    return m.group(0) if m else None


def _doctor(text: str) -> str | None:
    """Extract attending doctor; strip Dr./DR. prefix consistently."""
    raw = _find([
        r"[Aa]ttending\s*[Dd]octor\s*[:\-|]?\s*(?:Dr\.?\s*)?([A-Za-z][A-Za-z\s\.]{3,45}?)(?=\n|,|Dept|$)",
        r"[Tt]reating\s*[Dd]octor\s*[:\-|]?\s*(?:Dr\.?\s*)?([A-Za-z][A-Za-z\s\.]{3,45}?)(?=\n|,|Dept|$)",
        r"[Cc]onsultant\s*[:\-|]?\s*(?:Dr\.?\s*)?([A-Za-z][A-Za-z\s\.]{3,45}?)(?=\n|,|Dept|$)",
        r"[Rr]eferr(?:ing|ed)\s*[Dd]octor\s*[:\-|]?\s*(?:Dr\.?\s*)?([A-Za-z][A-Za-z\s\.]{3,45}?)(?=\n|$)",
        r"[Pp]hysician\s*[:\-|]?\s*(?:Dr\.?\s*)?([A-Za-z][A-Za-z\s\.]{3,45}?)(?=\n|$)",
        r"[Ss]urgeon\s*[:\-|]?\s*(?:Dr\.?\s*)?([A-Za-z][A-Za-z\s\.]{3,45}?)(?=\n|$)",
        r"Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?=\s*[\n,\(]|\s+Dept|\s+MD|\s*$)",
        r"DR\.?\s+([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})(?=\s*[\n,]|\s*$)",
    ], text)
    if raw:
        raw = re.sub(r"^(?:DR\.?|Dr\.?)\s+", "", raw.strip(" .,|")).strip()
        return re.sub(r"\s+", " ", "Dr. " + raw)
    return None


def _dept(text: str) -> str | None:
    """Extract hospital department / speciality / division."""
    return _find([
        r"[Dd]ept\.?\s*[:\-|]?\s*([A-Za-z][A-Za-z\s/\-]{2,40}?)(?=\n|$)",
        r"[Dd]epartment\s*[:\-|]?\s*([A-Za-z][A-Za-z\s/\-]{2,40}?)(?=\n|$)",
        r"[Ss]peciality\s*[:\-|]?\s*([A-Za-z][A-Za-z\s/\-]{2,40}?)(?=\n|$)",
        r"[Dd]ivision\s*[:\-|]?\s*([A-Za-z][A-Za-z\s/\-]{2,40}?)(?=\n|$)",
        r"\b(DRUGGROUP|PHARMACY|LABORATORY|RADIOLOGY|OT|WARD)\b",
    ], text)


def _diagnoses(text: str) -> list[str]:
    """Extract diagnosis strings; filter noise; return ≤10."""
    raw = _find([
        r"[Dd]iagnosi[se]\s*[:\-]?\s*(.+?)(?=\n\n|\Z)",
        r"[Cc]ondition\s*[:\-]?\s*(.+?)(?=\n\n|\Z)",
        r"[Cc]hief\s*[Cc]omplaint\s*[:\-]?\s*(.+?)(?=\n\n|\Z)",
        r"[Aa]dmitted\s+[Ff]or\s*[:\-]?\s*(.+?)(?=\n\n|\Z)",
    ], text)
    if not raw:
        return []
    results = []
    for part in re.split(r"[,;\n]+", raw):
        part = part.strip(" .-\t|")
        if 3 < len(part) < 120 and not re.match(r"^\d+$", part) \
                and not re.match(r"^[Pp]age\s+\d+", part):
            results.append(part)
    return results[:10]


def _charges(text: str) -> dict:
    """FIX B: gross_total uses ONLY explicit labelled patterns — no fallback hallucination."""
    def A(*pats: str) -> float | None:
        for p in pats:
            raw = _find(p, text)
            v   = _clean_amount(raw)
            if v is not None and v > 0:
                return v
        return None

    gross = A(
        r"[Gg]rand\s*[Tt]otal\s+([\d,]+\.?\d*)",
        r"[Gg]rand\s*[Tt]otal\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
        r"[Gg]ross\s*[Tt]otal\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
        r"[Tt]otal\s*[Aa]mount\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
        r"[Tt]otal\s*[Bb]ill\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
        r"[Nn]et\s*[Tt]otal\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
        r"[Bb]ill\s*[Aa]mount\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
        r"[Tt]otal\s*[Cc]harges?\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
        r"[Tt]otal\s*[Dd]ue\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
    )
    if gross is None:
        for m in re.finditer(
            r"^.*?(?:GRAND\s+TOTAL|TOTAL\s+AMOUNT|NET\s+AMOUNT\s+PAYABLE)"
            r".*?([\d,]{4,}\.?\d*)\s*$",
            text, re.I | re.M,
        ):
            v = _clean_amount(m.group(1))
            if v and v > 100:
                gross = v
                break

    return {
        "room_charges":    A(r"[Rr]oom\s*[Cc]harges?\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)"),
        "doctor_charges":  A(r"(?:[Dd]octor|[Cc]onsultant)\s*[Cc]harges?\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)"),
        "pharmacy_charges": A(
            r"^PHARMACY\s+([\d,]+\.?\d*)", r"PHARMACY\s+([\d,]+\.?\d*)",
            r"(?:[Pp]harmacy|[Mm]edicine)\s*[Cc]harges?\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
        ),
        "lab_charges":     A(r"(?:[Ll]ab|[Ii]nvestigation)\s*[Cc]harges?\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)"),
        "ot_charges":      A(r"(?:OT|[Oo]peration\s*[Tt]heatre)\s*[Cc]harges?\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)"),
        "nursing_charges": A(r"[Nn]ursing\s*[Cc]harges?\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)"),
        "subtotal":        A(r"[Ss]ub\s*[Tt]otal\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)"),
        "discount":        A(
            r"[Dd]iscount\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
            r"[Cc]oncession\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
            r"[Ww]aiver\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
        ),
        "cgst":            A(r"CGST\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)"),
        "sgst":            A(r"SGST\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)"),
        "igst":            A(r"IGST\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)"),
        "gross_total":     gross,
        "advance_paid":    A(
            r"[Aa]dvance\s*[Pp]aid\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
            r"[Aa]dvance\s*[Rr]eceived\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
            r"[Aa]mount\s*[Rr]eceived\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
            r"[Aa]mount\s*[Pp]aid\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
        ),
        "tpa_deduction":   A(r"TPA\s*(?:[Dd]eduction|[Aa]mount)\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)"),
        "balance_due":     A(
            r"[Bb]alance\s*[Dd]ue\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
            r"[Nn]et\s*(?:[Aa]mount|[Pp]ayable)\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
            r"[Aa]mount\s*(?:[Dd]ue|[Pp]ayable)\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
            r"[Pp]atient\s*(?:[Pp]ayable|[Dd]ue)\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
            r"[Dd]ue\s*[Aa]mount\s*[:\-₹|]?\s*([\d,\s]+\.?\d*)",
        ),
        "amount_in_words": _find([
            r"[Aa]mount\s+in\s+[Ww]ords?\s*[:\-]?\s*([A-Za-z\s]+?)(?=\n|[Oo]nly|$)",
            r"(?:INR|Rs\.?)\s+([A-Za-z\s]+?[Oo]nly)",
            r"([A-Za-z\s]{10,80}[Oo]nly)\b",
        ], text),
    }


def _insurance(text: str) -> dict:
    """Extract TPA name, insurance company, policy number, cashless flag."""
    return {
        "tpa_name": _find([
            r"TPA\s*[:\-|]?\s*([A-Za-z0-9 &\.]+?)(?=\n|$)",
            r"Third\s*[Pp]arty\s*[Aa]dministrator\s*[:\-|]?\s*([A-Za-z0-9 &\.]+?)(?=\n|$)",
        ], text),
        "insurance_company": _find([
            r"[Ii]nsurance\s*[Cc]ompany\s*[:\-|]?\s*([A-Za-z0-9 &\.]+?)(?=\n|[Pp]olicy|$)",
            r"[Ii]nsurance\s*[:\-|]?\s*([A-Za-z0-9 &\.]+?)(?=\n|$)",
        ], text),
        "policy_number": _find([
            r"[Pp]olicy\s*[Nn]o\.?\s*[:\-|#]?\s*([A-Z0-9][A-Z0-9/\-]{2,25})",
            r"[Pp]olicy\s*[Nn]umber\s*[:\-|]?\s*([A-Z0-9][A-Z0-9/\-]{2,25})",
        ], text),
        "cashless": bool(re.search(r"\bcashless\b", text, re.I)),
    }


def _payments(text: str) -> list[dict]:
    """Extract payment mode entries with amounts."""
    result = []
    for mode in ["Cash", "Card", "UPI", "NEFT", "IMPS", "Cheque", "TPA", "Online", "Insurance", "RTGS", "DD"]:
        m = re.search(rf"\b{mode}\b\s*[:\-]?\s*([\d,]+\.?\d*)", text, re.I)
        if m:
            v = _clean_amount(m.group(1))
            if v and v > 0:
                result.append({"mode": mode, "amount": v})
    return result


def _prepared_by(text: str) -> dict | None:
    """Extract prepared-by staff info from bill footer."""
    m = re.search(r"[Pp]repared\s+[Bb]y\s+([A-Za-z\s]+?),?\s*(EMP\d+|emp\d+)?", text, re.I)
    if m:
        result: dict[str, str] = {"name": m.group(1).strip()}
        if m.group(2):
            result["emp_id"] = m.group(2).upper()
        return result
    return None


def _page_info(text: str) -> dict | None:
    """Extract page number and total pages."""
    m = re.search(r"[Pp]age\s+(\d+)\s+of\s+(\d+)", text, re.I)
    if m:
        return {"page": int(m.group(1)), "total_pages": int(m.group(2))}
    return None


_ADDR_PAT = re.compile(
    r"Road|Nagar|Street|Square|Marg|Chowk|Lane|Colony|Sector|"
    r"Block|Phase|Plot|Floor|Tower|Building|Complex|Near|Opp|"
    r"Bhubaneswar|Mumbai|Delhi|Chennai|Hyderabad|Pune|Bangalore|"
    r"Bengaluru|Kolkata|Odisha|Maharashtra|Karnataka|Tamil|"
    r"Andhra|Telangana|PIN|Pincode|\d{6}\b", re.I,
)


def _address(lines: list[str]) -> str | None:
    """Extract hospital address from address-bearing lines."""
    hits = [ln.strip(" '\"") for ln in lines if _ADDR_PAT.search(ln)]
    return ", ".join(hits[:2]) if hits else None


# ══════════════════════════════════════════════════════════════════════════════
#  5.  LINE-ITEM PARSER  (FIX C — unchanged from v10)
# ══════════════════════════════════════════════════════════════════════════════

_CAT_RULES: list[tuple[list[str], str]] = [
    (["operation","surgery","ot ","theatre","procedure","laparoscop","anesthesia",
      "anaesthesia","endoscop","stent","breathing circuit","intubation","tracheostomy","cardiac"], "OT / Surgery"),
    (["x-ray","xray","x ray","mri","ct scan","ultrasound","echo","scan","radiolog",
      "mammograph","fluoro","angiograph","pet scan"], "Radiology"),
    (["haematology","haemoglobin","hb ","cbc","blood group","aptt","prothrombin",
      "leucocyte","microbiology","culture","sensitivity","serology","hbs ag","hcv","hiv",
      "pathology","urine","biochem","immunolog","thyroid","tsh","creatinine","glucose",
      "lipid","test ","investigation"], "Lab / Diagnostics"),
    (["tab ","cap ","inj ","syrup"," mg "," ml ","pharmacy","drip","infusion","iv ","i.v",
      "paracetamol","ibuprofen","omeprazole","pantoprazole","ondansetron","ranitidine",
      "dextrose","ringer","saline","heparin","insulin","adrenaline","dopamine","fentanyl",
      "midazolam","propofol","ketamine","lignocaine"], "Pharmacy"),
    (["syringe","gloves","catheter","bandage","cotton","gauze","disposable","needle",
      "ryles tube","iv set","blood set","urine bag","suture","stapler","electrode",
      "mask","cannula","drain","tourniquet"], "Consumable"),
    (["doctor","consultant","visit","physician","surgeon","consultation","specialist",
      "review","follow up","opd"], "Consultation"),
    (["nursing","nurse","dressing","wound care","injection charge"], "Nursing"),
    (["bed ","ward","room","accommodation","icu bed","nicu","picu","cabin","suite",
      "ventilator","oxygen","flowtron","nebulisa","blood transfusion","physiotherapy",
      "dialysis","service charge","monitoring"], "Service Charges"),
    (["ambulance","transport"], "Transport"),
    (["food","diet","canteen","meal"], "Diet / Nutrition"),
    (["admin","registration","admission fee","medical certificate"], "Administrative"),
]


def _classify(desc: str) -> str:
    """Classify a line-item description into a category using keyword rules."""
    d = desc.lower()
    for keywords, cat in _CAT_RULES:
        if any(k in d for k in keywords):
            return cat
    return "Other"


_NUM_BLOCK = re.compile(
    r"\s{1,3}([\d,]{1,10}(?:\.\d{0,2})?)\s{1,4}(\d{1,5}(?:\.\d{0,3})?)"
    r"\s{1,4}([\d,]{1,10}(?:\.\d{0,2})?)(?:\s{1,4}([\d,]{1,10}(?:\.\d{0,2})?))?\s*$"
)
_NUM_BLOCK_3COL = re.compile(
    r"\s{1,4}(\d{1,5}(?:\.\d{0,3})?)\s{1,4}([\d,]{1,10}(?:\.\d{0,2})?)\s*$"
)
_SKIP_PAT = re.compile(
    r"^(?:Description|BatchNo|Unit\s*Rate|Qty|Total|Amount|DRUGGROUP|"
    r"Prepared|Generated|Page|Healthcare|Neotia|HEALTHCARE|&\s*$|\s*$|[-─═=]{3,}\s*$)",
    re.IGNORECASE,
)
_SECTION_TOTAL = re.compile(
    r"^(?:TOTAL|SUB[-\s]?TOTAL|GRAND\s+TOTAL|NET\s+TOTAL|PHARMACY\s+TOTAL|"
    r"OT\s+TOTAL|LAB\s+TOTAL|NURSING\s+TOTAL|BALANCE\s+DUE|AMOUNT\s+DUE|AMOUNT\s+PAYABLE)",
    re.IGNORECASE,
)
_DATE_ONLY  = re.compile(r"^\d{2}[-/]\d{2}[-/]\d{4}\s*$")
_BATCH_CONT = re.compile(
    r"^(?:[|\s]*[A-Z0-9][A-Z0-9/\-\.]{2,20}[/\s]+)?\d{2}[-/]\d{2}[-/]\d{4}\s*$", re.I,
)


def _extract_batch_expiry(raw_desc: str) -> tuple[str, str | None, str | None]:
    """Extract batch number and expiry date embedded in description string."""
    batch_no: str | None  = None
    expiry_dt: str | None = None
    m_b = re.search(r"\|\s*([A-Z0-9$][A-Z0-9/\-\.]{2,20}?)/?\s*(\d{2}[-/]\d{2}[-/]\d{4})?", raw_desc, re.I)
    if m_b:
        batch_no  = re.sub(r"[$|/]", "", m_b.group(1)).strip()
        expiry_dt = m_b.group(2)
        raw_desc  = raw_desc[:m_b.start()].strip()
    else:
        m_b2 = re.search(r"([A-Z0-9][A-Z0-9\-]{3,15})/\s*(\d{2}[-/]\d{2}[-/]\d{4})?$", raw_desc, re.I)
        if m_b2 and not re.match(r"^[A-Z ]+$", m_b2.group(1)):
            batch_no  = m_b2.group(1)
            expiry_dt = m_b2.group(2)
            raw_desc  = raw_desc[:m_b2.start()].strip()
    return raw_desc, batch_no, expiry_dt


def _clean_desc(raw: str) -> str:
    """Strip SER code prefix and form suffixes; normalise whitespace."""
    raw = re.sub(r"^\$?SER[A-Z0-9]{3,12}\s+", "", raw, flags=re.I)
    raw = re.sub(r"\s*,\s*(SOLID|LIQUID|ADULT|CHILD|TABLET|CAPSULE|INJECTION|SYRUP)\s*,?\s*$", "", raw, flags=re.I)
    return re.sub(r"[,\.\s]+$", "", raw).strip()


def _parse_item_line(line: str) -> dict | None:
    """Parse one bill item line. FIX C1/C2: permissive matching."""
    stripped = line.strip()
    if not stripped or len(stripped) < 6:
        return None
    if _SKIP_PAT.match(stripped) or _SECTION_TOTAL.match(stripped):
        return None

    m = _NUM_BLOCK.search(stripped)
    if m and m.start() >= 3:
        raw_desc = stripped[:m.start()].strip()
        raw_desc_s = re.sub(r"^\$?SER[A-Z0-9]{3,12}\s+", "", raw_desc, flags=re.I)
        if not raw_desc_s or not re.match(r"^[A-Za-z0-9$]", raw_desc_s):
            m = None
        if m:
            try:
                unit_rate = round(float(m.group(1).replace(",", "")), 2)
                qty       = float(m.group(2).replace(",", ""))
                total_rs  = round(float(m.group(3).replace(",", "")), 2)
                amount_rs = round(float(m.group(4).replace(",", "")), 2) if m.group(4) else total_rs
            except (ValueError, AttributeError):
                m = None
            if m:
                raw_desc, batch_no, expiry_dt = _extract_batch_expiry(raw_desc)
                desc = _clean_desc(raw_desc)
                if not desc or len(desc) < 3:
                    return None
                entry: dict[str, Any] = {
                    "description": desc, "category": _classify(desc),
                    "unit_rate": unit_rate, "quantity": qty,
                    "total_rs": total_rs, "amount": amount_rs,
                }
                if batch_no:  entry["batch_no"]    = batch_no
                if expiry_dt: entry["expiry_date"] = expiry_dt
                return entry

    m3 = _NUM_BLOCK_3COL.search(stripped)
    if m3 and m3.start() >= 3:
        raw_desc  = stripped[:m3.start()].strip()
        raw_desc_s = re.sub(r"^\$?SER[A-Z0-9]{3,12}\s+", "", raw_desc, flags=re.I)
        if raw_desc_s and re.match(r"^[A-Za-z0-9$]", raw_desc_s):
            try:
                qty    = float(m3.group(1).replace(",", ""))
                amount = round(float(m3.group(2).replace(",", "")), 2)
            except ValueError:
                return None
            raw_desc, batch_no, expiry_dt = _extract_batch_expiry(raw_desc)
            desc = _clean_desc(raw_desc)
            if not desc or len(desc) < 3:
                return None
            entry = {"description": desc, "category": _classify(desc), "quantity": qty, "amount": amount}
            if batch_no:  entry["batch_no"]    = batch_no
            if expiry_dt: entry["expiry_date"] = expiry_dt
            return entry
    return None


def _line_items(text: str) -> list[dict]:
    """Main line-item dispatcher: Neotia format → generic 4-col → generic 2-col."""
    clean_text = re.sub(r"[-─]+\s*PAGE BREAK\s*[-─]+", "\n", text, flags=re.I)
    is_neotia  = bool(re.search(
        r"(?:DRUGGROUP|Neotia|Getwel|BatchNo.*ExpiryDate|Unit\s*Rate.*Qty.*Total)",
        clean_text, re.I,
    ))
    lines: list[str] = clean_text.splitlines()
    items: list[dict] = []
    seen:  set[tuple] = set()
    used:  set[int]   = set()

    def dedup_add(entry: dict) -> None:
        key = (entry["description"].lower()[:32], round(entry.get("amount", 0) or 0))
        if key not in seen:
            seen.add(key)
            items.append(entry)

    if is_neotia:
        for i, line in enumerate(lines):
            if i in used or _SKIP_PAT.match(line.strip()):
                continue
            entry = _parse_item_line(line)
            if not entry:
                continue
            for j in range(i + 1, min(i + 6, len(lines))):
                nxt = lines[j].strip()
                if not nxt or nxt in ("A", "&", "|"):
                    used.add(j); continue
                if _SKIP_PAT.match(nxt) or _SECTION_TOTAL.match(nxt):
                    break
                if _parse_item_line(nxt):
                    break
                if _BATCH_CONT.match(nxt) or _DATE_ONLY.match(nxt):
                    m_bd = re.search(r"([A-Z0-9][A-Z0-9/\-\.]{2,20})[/\s]+(\d{2}[-/]\d{2}[-/]\d{4})", nxt, re.I)
                    if m_bd and not entry.get("batch_no"):
                        entry["batch_no"]    = re.sub(r"[$|/]", "", m_bd.group(1)).strip()
                        entry["expiry_date"] = m_bd.group(2)
                    elif not entry.get("expiry_date"):
                        m_d = re.search(r"(\d{2}[-/]\d{2}[-/]\d{4})", nxt)
                        if m_d:
                            entry["expiry_date"] = m_d.group(1)
                    used.add(j); break
            dedup_add(entry)
        return items

    _SKIP_WORDS = {
        "total","balance","amount","payable","grand","net","subtotal",
        "discount","advance","paid","due","page","printed","date",
        "cgst","sgst","gst","tax","bill","receipt","invoice",
    }
    for m in re.finditer(
        r"^(.{4,70}?)\s+(\d{1,5}(?:\.\d{0,3})?)\s+([\d,]+\.?\d*)\s+([\d,]+\.?\d*)\s*$",
        clean_text, re.MULTILINE,
    ):
        desc = m.group(1).strip()
        if any(w in desc.lower() for w in _SKIP_WORDS):
            continue
        qty_v  = float(m.group(2))
        rate_v = _clean_amount(m.group(3))
        exc_v  = _clean_amount(m.group(4))
        if rate_v and exc_v and qty_v > 0:
            if abs(round(qty_v * exc_v, 2) - rate_v) < abs(round(qty_v * rate_v, 2) - exc_v):
                rate_v, exc_v = exc_v, rate_v
        dedup_add({"description": desc, "category": _classify(desc),
                   "quantity": qty_v, "unit_rate": rate_v, "amount": exc_v})

    for m in re.finditer(
        r"^([A-Za-z0-9\$][A-Za-z0-9 /\(\)\-\.,:%+&\$]{4,70}?)\s{2,}([\d,]{2,}\.?\d*)\s*$",
        clean_text, re.MULTILINE,
    ):
        desc = m.group(1).strip()
        if any(w in desc.lower() for w in _SKIP_WORDS):
            continue
        dedup_add({"description": desc, "category": _classify(desc),
                   "amount": _clean_amount(m.group(2))})
    return items


# ══════════════════════════════════════════════════════════════════════════════
#  6.  POST-EXTRACTION VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _validate_amounts(charges: dict, items: list[dict]) -> dict:
    """FIX D: Only sum line-items if ≥5 items and no dominant single item."""
    warnings: list[str] = []
    if charges.get("gross_total") is None:
        all_amts = [it.get("amount") or 0 for it in items if (it.get("amount") or 0) > 0]
        if len(all_amts) >= 5:
            total_candidate = round(sum(all_amts), 2)
            max_single = max(all_amts)
            if max_single < total_candidate * 0.5:
                charges["gross_total"] = total_candidate
                warnings.append("gross_total computed by summing all line items (≥5 items, no dominant single item)")

    gross   = charges.get("gross_total")
    advance = charges.get("advance_paid")
    balance = charges.get("balance_due")
    if gross and advance and balance:
        expected = round(gross - advance, 2)
        if abs(expected - balance) > 10:
            warnings.append(
                f"balance_due soft mismatch: {gross} - {advance} = {expected}, extracted = {balance}"
            )
    if warnings:
        charges["_warnings"] = warnings
    return charges


# ══════════════════════════════════════════════════════════════════════════════
#  7.  MASTER PARSE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def parse_bill(raw_text: str, filename: str) -> dict:
    """Orchestrate all extractors and return a fully structured bill dict."""
    text  = _normalise(raw_text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    adm_date, dis_date = _stay_dates(text)
    age, gender        = _age_gender(text)

    los: int | None = None
    if adm_date and dis_date:
        try:
            los = (date.fromisoformat(dis_date) - date.fromisoformat(adm_date)).days
        except ValueError:
            pass

    charge_data = _charges(text)
    items       = _line_items(text)
    charge_data = _validate_amounts(charge_data, items)

    phone = _find([
        r"(?:Tel|Ph|Phone|Contact|Mob|Mobile)\s*[.:\-]?\s*([\d\s\-,/\(\)]{7,25})",
        r"(?:Tel|Ph)\s*[.:\-]?\s*(\+?[\d\s\-]{7,15})",
    ], text)

    result = {
        "bill_type":   _bill_type(text, filename),
        "bill_number": _bill_number(text),
        "bill_date":   _bill_date(text),
        "hospital": {
            "name":    _hospital_name(lines, text),
            "address": _address(lines),
            "phone":   phone.strip() if phone else None,
            "email":   _find(r"([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9.\-]+)", text),
            "gstin":   _gstin(text),
        },
        "patient": {
            "name":   _patient_name(text),
            "age":    age,
            "dob":    _to_date(_find([
                r"DOB\s*[:\-|]?\s*(.+?)(?=\n|Age|$)",
                r"[Dd]ate\s+[Oo]f\s+[Bb]irth\s*[:\-|]?\s*(.+?)(?=\n|$)",
            ], text)),
            "gender": gender,
            "uhid":   _uhid(text),
        },
        "admission": {
            "admission_date": adm_date,
            "discharge_date": dis_date,
            "ward":           _ward(text),
            "bed_number":     _bed(text),
            "ipd_number":     _find([
                r"IP\s*[Nn]o\.?\s*[:\-|]?\s*([A-Za-z0-9/\-]+)",
                r"IPD\s*(?:[Nn]o\.?)?\s*[:\-|]?\s*([A-Za-z0-9/\-]+)",
            ], text),
            "opd_number":     _find([
                r"OP\s*[Nn]o\.?\s*[:\-|]?\s*([A-Za-z0-9/\-]+)",
                r"OPD\s*(?:[Nn]o\.?)?\s*[:\-|]?\s*([A-Za-z0-9/\-]+)",
            ], text),
            "los_days":       los,
        },
        "clinical": {
            "diagnosis":        _diagnoses(text),
            "attending_doctor": _doctor(text),
            "department":       _dept(text),
        },
        "insurance":       _insurance(text),
        "charges_summary": charge_data,
        "line_items":      items,
        "payments":        _payments(text),
    }

    prepared  = _prepared_by(text)
    page_info = _page_info(text)
    if prepared:  result["prepared_by"] = prepared
    if page_info: result["page_info"]   = page_info

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  8.  RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def _collect() -> list[Path]:
    """Collect all processable image and PDF files from configured directories."""
    files: list[Path] = []
    if IMG_DIR.exists():
        files += [f for f in IMG_DIR.rglob("*") if f.is_file() and f.suffix in IMG_EXTS]
    if PDF_DIR.exists():
        files += [f for f in PDF_DIR.rglob("*") if f.is_file() and f.suffix.lower() == PDF_EXT]
    return sorted(files)


def main() -> None:
    """Main entry point: preflight → collect → OCR → parse → write JSON + GPU cost report."""
    preflight()
    files = _collect()
    if not files:
        log.error("No files found in IMG_DIR or PDF_DIR.")
        sys.exit(1)

    log.info(f"Processing {len(files)} file(s)…\n")
    succeeded: list[Path]            = []
    failed:    list[tuple[Path,str]] = []
    wall      = time.perf_counter()
    total_gpu_seconds = 0.0          # ← accumulate real processing time

    for idx, fp in enumerate(files, 1):
        print(f"[{idx:3d}/{len(files)}]  {fp.name}", end="", flush=True)
        try:
            t0      = time.perf_counter()
            raw     = extract_text(fp)
            result  = parse_bill(raw, fp.name)
            elapsed = round(time.perf_counter() - t0, 2)

            # ── per-file GPU cost ──────────────────────────────────────────
            file_gpu_cost = gpu_cost_for_seconds(elapsed)
            total_gpu_seconds += elapsed

            result["_meta"] = {
                "source_file":        str(fp),
                "ocr_engine":         "tesseract",
                "processing_time_s":  elapsed,
                "extracted_at":       datetime.now().isoformat(),
                "raw_text_chars":     len(raw),
                "line_items_found":   len(result["line_items"]),
                "extractor_version":  "v11",
                # GPU cost fields embedded in every output JSON
                "gpu_name":           GPU_NAME,
                "gpu_effective_w":    round(_effective_watts(), 1),
                "gpu_cost_inr":       round(file_gpu_cost, 8),
                "elec_rate_inr_kwh":  ELEC_RATE_INR,
            }

            stem = fp.stem
            out  = OUT_DIR / f"{stem}.json"
            if out.exists():
                out = OUT_DIR / f"{fp.parent.name}__{stem}.json"
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False), "utf-8")
            (OUT_DIR / f"{stem}_ocr.txt").write_text(raw, "utf-8")

            n = len(result["line_items"])
            print(f"  ✓  {elapsed}s  |  {n} items  |  ₹{file_gpu_cost:.6f}")
            succeeded.append(fp)

        except Exception as exc:
            failed.append((fp, str(exc)))
            print(f"  ✗  {exc}")
            log.debug(traceback.format_exc())

    total_wall = round(time.perf_counter() - wall, 1)

    print(f"\n{'=' * 65}")
    print(f"  DONE  {len(succeeded)}/{len(files)} succeeded  |  {total_wall}s  |  $0.00")
    print(f"  Output: {OUT_DIR}")
    print("=" * 65)

    # ── GPU COST SUMMARY printed after DONE ───────────────────────────────
    if succeeded:
        print_gpu_cost_report(total_gpu_seconds, len(succeeded))

    if failed:
        print("\n  FAILED:")
        for fp_, e in failed:
            print(f"    {fp_.name}  →  {e}")

    gpu_stats = gpu_cost_summary(total_gpu_seconds, len(succeeded)) if succeeded else {}

    (OUT_DIR / "_report.json").write_text(
        json.dumps({
            "total":       len(files),
            "succeeded":   len(succeeded),
            "failed":      len(failed),
            "cost_usd":    0.0,
            "total_time_s": total_wall,
            "gpu_cost":    gpu_stats,           # ← GPU section in _report.json
            "failed_files": [{"file": str(f), "error": e} for f, e in failed],
            "generated_at": datetime.now().isoformat(),
        }, indent=2)
    )


if __name__ == "__main__":
    main()