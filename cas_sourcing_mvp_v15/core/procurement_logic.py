from __future__ import annotations

import re
import pandas as pd

PRICE_HIDDEN = "No public price detected"

PRICE_NOISE_RE = re.compile(
    r"(?i)(free\s+shipping|orders?\s+over|minimum\s+order|cart\s*(?:total|subtotal)|basket\b|subtotal|checkout|coupon|promo|discount|shipping\s+threshold|handling\s+fee|tax\b|recently\s+added|sign\s+in\s+to\s+checkout)"
)
SEARCH_URL_RE = re.compile(r"(?i)(/search|catalogsearch|keyword=|search=|q=|query=|find\.cgi|Search\?|result/\?)")
PRODUCT_HINT_RE = re.compile(r"(?i)(/product|/products/|/compound/|/item/|/catalog/|/p/|/pd/|\.html$|\.htm$)")
STANDARD_RE = re.compile(r"(?i)(\bstandard\b|reference\s+standard|analytical\s+standard|certified\s+reference)")
SOLUTION_RE = re.compile(r"(?i)(\bmM\b|\bin\s*DMSO\b|solution|assay-ready|assay\s+ready|stock\s+solution)")
PURITY_VALUE_RE = re.compile(r"(?P<value>\d{1,3}(?:\.\d+)?)\s*%")
PURITY_CONTEXT_NOISE_RE = re.compile(
    r"(?i)(happy\s+customers|discount|coupon|promo|off\b|save\b|free\s+shipping|orders?\s+over|shipping|cart total|subtotal|tax|rating|reviews?|cell\s+viability|inhibition|recovery|dmso|\bmM\b)"
)


def parse_required_purity(required_purity: str | None) -> float | None:
    if not required_purity:
        return None
    m = PURITY_VALUE_RE.search(str(required_purity))
    if not m:
        return None
    try:
        value = float(m.group("value"))
    except ValueError:
        return None
    return value if 0 < value <= 100 else None


def parse_purity_value(purity: str | None) -> float | None:
    if purity is None:
        return None
    text = str(purity)
    if not text or text.lower() in {"nan", "none", "not visible"}:
        return None
    m = PURITY_VALUE_RE.search(text)
    if not m:
        return None
    try:
        value = float(m.group("value"))
    except ValueError:
        return None
    return value if 0 < value <= 100 else None


def classify_page_type(url: str | None, title: str | None, extraction_status: str | None = None) -> str:
    hay = f"{url or ''} {title or ''}".lower()
    if extraction_status and str(extraction_status).startswith("failed"):
        return "failed_source"
    if SEARCH_URL_RE.search(hay) or "search results" in hay or "résultats" in hay or "검색" in hay:
        return "search_page"
    if PRODUCT_HINT_RE.search(str(url or "")):
        return "product_page"
    return "unknown_page"


def classify_product_form(row: pd.Series) -> str:
    title = str(row.get("page_title") or row.get("title") or "")
    url = str(row.get("product_url") or "")
    raw = str(row.get("raw_matches") or "")
    notes = str(row.get("notes") or "")
    pack_unit = str(row.get("pack_unit") or "").strip()
    hay = f"{title} {url} {raw} {notes}"
    if STANDARD_RE.search(hay):
        return "reference_standard"
    if pack_unit in {"mL", "L", "ml", "l"}:
        return "liquid_or_solution"
    if pack_unit in {"mg", "g", "kg", "ug", "µg", "μg"}:
        if re.search(r"\b\d+(?:\.\d+)?\s?m[lL]\s*(?:x|\*)?\s*\d+(?:\.\d+)?\s?mM\b|\b\d+(?:\.\d+)?\s?mM\s*(?:in\s*DMSO|x|\*)", raw, re.I):
            return "assay_solution"
        return "mass_catalog"
    if SOLUTION_RE.search(hay):
        return "assay_solution"
    return "unknown_form"


def price_context_is_noise(row: pd.Series) -> bool:
    if pd.isna(row.get("listed_price_usd")):
        return False
    hay = " ".join(str(row.get(col) or "") for col in ["raw_matches", "notes", "page_title", "product_url", "stock_status"])
    return bool(PRICE_NOISE_RE.search(hay))


def purity_context_is_noise(row: pd.Series) -> bool:
    if row.get("purity_value_pct") is None or pd.isna(row.get("purity_value_pct")):
        return False
    hay = " ".join(str(row.get(col) or "") for col in ["raw_matches", "notes", "page_title", "product_url", "stock_status"])
    return bool(PURITY_CONTEXT_NOISE_RE.search(hay))


def _purity_pass_label(purity_value: float | None, required_threshold: float | None) -> str:
    if required_threshold is None:
        return "Not required"
    if purity_value is None:
        return "Unknown"
    return "Yes" if purity_value >= required_threshold else "No"


def _trust_decision(row: pd.Series) -> str:
    if bool(row.get("price_noise_flag")):
        return "Reject price: non-product price context"
    page_type = str(row.get("page_type") or "")
    product_form = str(row.get("product_form") or "")
    pairing = str(row.get("price_pairing_confidence") or "")
    purity_pass = str(row.get("purity_pass") or "")
    has_price = pd.notna(row.get("listed_price_usd"))
    if bool(row.get("tentative_price")):
        return "Candidate price only: verify product/CAS before use"
    if page_type == "search_page":
        return "Discovery only: not a product page"
    if page_type == "failed_source":
        return "Manual review: extraction failed"
    if product_form in {"assay_solution", "reference_standard", "liquid_or_solution"}:
        return "Catalog evidence only: not bulk mass-estimate eligible"
    if purity_pass == "No":
        return "Downgrade: below required purity"
    if has_price and pairing in {"HIGH", "MEDIUM"}:
        return "Use for catalog comparison and curve model"
    if has_price:
        return "Show price, but verify pairing manually"
    return "Supplier lead / RFQ candidate"


def _trust_warning(row: pd.Series) -> str:
    warnings: list[str] = []
    if bool(row.get("price_noise_flag")):
        warnings.append("price looked like shipping/cart/promo noise")
    if str(row.get("purity_confidence")) == "REJECTED":
        warnings.append("purity-like percentage looked like marketing/assay-solution noise")
    if bool(row.get("tentative_price")):
        warnings.append("price retained as low-confidence lead; verify product page and CAS before comparing")
    if str(row.get("page_type")) == "search_page":
        warnings.append("source is a search/results page")
    if str(row.get("product_form")) == "assay_solution":
        warnings.append("solution/DMSO format is not comparable to solid mg/g pricing")
    if str(row.get("product_form")) == "reference_standard":
        warnings.append("reference standard/product subtype should not be mixed with bulk material")
    if str(row.get("purity_pass")) == "No":
        warnings.append("purity is below requested threshold")
    if str(row.get("purity_pass")) == "Unknown" and row.get("purity_required_threshold") is not None:
        warnings.append("purity not visible; verify before sourcing")
    return "; ".join(warnings) or ""


def enrich_procurement_trust(df: pd.DataFrame, required_purity: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()
    out = df.copy()
    threshold = parse_required_purity(required_purity)
    out["purity_required_threshold"] = threshold
    if "purity" not in out.columns:
        out["purity"] = None
    out["purity_value_pct"] = out["purity"].map(parse_purity_value)
    out["purity_confidence"] = out["purity_value_pct"].map(lambda v: "HIGH" if pd.notna(v) else "NONE")
    for col in ["product_url", "page_title", "extraction_status", "raw_matches", "notes", "stock_status", "price_pairing_confidence"]:
        if col not in out.columns:
            out[col] = ""
    purity_noise_mask = out.apply(purity_context_is_noise, axis=1)
    if purity_noise_mask.any():
        out.loc[purity_noise_mask, "purity_value_pct"] = None
        out.loc[purity_noise_mask, "purity_confidence"] = "REJECTED"
    out["purity_pass"] = out["purity_value_pct"].map(lambda v: _purity_pass_label(v if pd.notna(v) else None, threshold))
    for col in ["product_url", "page_title", "extraction_status", "raw_matches", "notes", "stock_status", "price_pairing_confidence"]:
        if col not in out.columns:
            out[col] = ""
    if "listed_price_usd" not in out.columns:
        out["listed_price_usd"] = None
    out["page_type"] = out.apply(lambda r: classify_page_type(r.get("product_url"), r.get("page_title"), r.get("extraction_status")), axis=1)
    out["product_form"] = out.apply(classify_product_form, axis=1)
    out["price_noise_flag"] = out.apply(price_context_is_noise, axis=1)
    noise_mask = out["price_noise_flag"].astype(bool)
    if noise_mask.any():
        out.loc[noise_mask, "listed_price_usd"] = None
        if "price_visibility_status" in out.columns:
            out.loc[noise_mask, "price_visibility_status"] = PRICE_HIDDEN
        if "best_action" in out.columns:
            out.loc[noise_mask, "best_action"] = "Ignore noisy price; open source or RFQ"

    has_price = out["listed_price_usd"].notna()
    has_cas = out.get("cas_exact_match", pd.Series([False] * len(out), index=out.index)).fillna(False).astype(bool)
    lead_cas = out.get("lead_cas_match", pd.Series([False] * len(out), index=out.index)).fillna(False).astype(bool)
    if "price_validation_level" not in out.columns:
        out["price_validation_level"] = "none"
    validation = out["price_validation_level"].fillna("none").astype(str).str.lower()
    pairing_ok = out["price_pairing_confidence"].astype(str).isin(["HIGH", "MEDIUM"])
    page_ok = out["page_type"].isin(["product_page", "unknown_page"])
    product_page_ok = out["page_type"].eq("product_page") | (out["page_type"].eq("unknown_page") & ~validation.str.contains("candidate|snippet|search|lead", regex=True))
    form_ok = out["product_form"].eq("mass_catalog")
    purity_ok = ~out["purity_pass"].eq("No")
    out["verified_public_price"] = has_price & has_cas & product_page_ok & pairing_ok & ~noise_mask & ~validation.str.contains("candidate|unconfirmed|search|lead", regex=True)
    out["tentative_price"] = has_price & ~out["verified_public_price"]
    out["lead_cas_match"] = lead_cas | (out["tentative_price"] & has_cas)
    out["price_validation_level"] = out["price_validation_level"].where(~out["verified_public_price"], "verified_product_price")
    out.loc[out["tentative_price"] & validation.str.contains("snippet"), "price_validation_level"] = "snippet_or_search_price"
    out.loc[out["tentative_price"] & ~validation.str.contains("snippet"), "price_validation_level"] = out.loc[out["tentative_price"] & ~validation.str.contains("snippet"), "price_validation_level"].replace({"none": "candidate_price"})
    out["bulk_estimate_eligible"] = out["verified_public_price"] & form_ok & purity_ok
    out["catalog_chart_eligible"] = out["verified_public_price"] & form_ok
    out["lead_price_chart_eligible"] = out["tentative_price"] & has_price
    out["procurement_trust_decision"] = out.apply(_trust_decision, axis=1)
    out["trust_warning"] = out.apply(_trust_warning, axis=1)
    return out
