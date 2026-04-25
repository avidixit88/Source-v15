from __future__ import annotations

import re
import pandas as pd

MASS_UNITS = {"ug", "µg", "mg", "g", "kg"}
REQUIRED_PURITY_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
PURITY_VALUE_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")


def parse_required_purity(required_purity: str | None) -> float | None:
    if not required_purity:
        return None
    m = REQUIRED_PURITY_RE.search(str(required_purity))
    if not m:
        return None
    try:
        val = float(m.group(1))
        return val if 0 < val <= 100 else None
    except Exception:
        return None


def parse_purity_value(purity: object) -> float | None:
    if purity is None or pd.isna(purity):
        return None
    text = str(purity)
    if text.lower() in {"nan", "none", "not visible", ""}:
        return None
    m = PURITY_VALUE_RE.search(text)
    if not m:
        return None
    try:
        val = float(m.group(1))
        return val if 0 < val <= 100 else None
    except Exception:
        return None


def infer_product_form(row: pd.Series) -> str:
    existing = row.get("product_form")
    if existing and str(existing).lower() not in {"nan", "none", "unknown"}:
        return str(existing)
    hay = " ".join(str(row.get(c, "")) for c in ["page_title", "raw_matches", "notes", "product_url"]).lower()
    unit = str(row.get("pack_unit", "")).strip()
    if "standard" in hay or "reference" in hay or "calibration" in hay:
        return "standard/reference"
    if "in dmso" in hay or "mm in" in hay or "solution" in hay or unit in {"mL", "L"}:
        return "solution"
    if unit.lower() in MASS_UNITS:
        return "solid/mass"
    return "unknown"


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def add_procurement_quality_columns(df: pd.DataFrame, required_purity: str | None = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    required_value = parse_required_purity(required_purity)
    out["required_purity_pct"] = required_value
    out["purity_value_pct"] = out.get("purity", pd.Series([None] * len(out))).apply(parse_purity_value)
    if "purity_confidence" not in out.columns:
        out["purity_confidence"] = out["purity_value_pct"].apply(lambda x: "MEDIUM" if pd.notna(x) else "NONE")
    out["product_form"] = out.apply(infer_product_form, axis=1)

    def purity_pass(row: pd.Series) -> str:
        if required_value is None:
            return "Not required"
        val = row.get("purity_value_pct")
        conf = str(row.get("purity_confidence", "")).upper()
        if pd.isna(val) or conf in {"REJECTED", "LOW"}:
            return "Unknown"
        return "Pass" if float(val) >= required_value else "Fail"

    out["purity_pass"] = out.apply(purity_pass, axis=1)
    out["mass_based_price"] = out.get("pack_unit", pd.Series([None] * len(out))).astype(str).str.lower().isin(MASS_UNITS)
    out["trusted_price_pair"] = out.get("price_pairing_confidence", pd.Series(["NONE"] * len(out))).astype(str).str.upper().isin({"HIGH", "MEDIUM"})
    out["cas_confirmed"] = out.get("cas_exact_match", pd.Series([False] * len(out))).apply(_truthy)
    out["public_price"] = out.get("listed_price_usd", pd.Series([None] * len(out))).notna()
    out["catalog_estimate_eligible"] = (
        out["cas_confirmed"]
        & out["public_price"]
        & out["trusted_price_pair"]
        & out["mass_based_price"]
        & out["product_form"].isin(["solid/mass", "unknown"])
        & ~out["purity_pass"].isin(["Fail"])
    )

    def flags(row: pd.Series) -> str:
        bits: list[str] = []
        if not row.get("cas_confirmed"):
            bits.append("CAS not confirmed")
        if row.get("product_form") not in {"solid/mass", "unknown"}:
            bits.append(f"{row.get('product_form')} product")
        if row.get("purity_pass") == "Fail":
            bits.append("below required purity")
        if row.get("purity_pass") == "Unknown" and required_value is not None:
            bits.append("purity unknown")
        if row.get("public_price") and not row.get("trusted_price_pair"):
            bits.append("price pair not proven")
        if not row.get("mass_based_price") and row.get("public_price"):
            bits.append("not mass based")
        return "; ".join(bits) if bits else "eligible catalog evidence"

    out["quality_flags"] = out.apply(flags, axis=1)
    return out
