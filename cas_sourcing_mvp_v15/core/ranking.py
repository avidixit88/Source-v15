from __future__ import annotations

import pandas as pd


def _as_bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)


def rank_supplier_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["score"] = 0
    if "cas_number" in out.columns:
        out.loc[out["cas_number"].notna(), "score"] += 15
    if "cas_exact_match" in out.columns:
        out.loc[_as_bool_series(out, "cas_exact_match"), "score"] += 30
    if "has_visible_price" in out.columns:
        out.loc[_as_bool_series(out, "has_visible_price"), "score"] += 15
    if "price_pairing_confidence" in out.columns:
        out.loc[out["price_pairing_confidence"].astype(str).eq("HIGH"), "score"] += 20
        out.loc[out["price_pairing_confidence"].astype(str).eq("MEDIUM"), "score"] += 10
    if "bulk_estimate_eligible" in out.columns:
        out.loc[_as_bool_series(out, "bulk_estimate_eligible"), "score"] += 25
    if "purity_pass" in out.columns:
        out.loc[out["purity_pass"].astype(str).eq("Yes"), "score"] += 12
        out.loc[out["purity_pass"].astype(str).eq("No"), "score"] -= 25
    elif "purity" in out.columns:
        out.loc[out["purity"].astype(str).str.contains("99|98|95", regex=True, na=False), "score"] += 10
    if "product_form" in out.columns:
        out.loc[out["product_form"].astype(str).eq("mass_catalog"), "score"] += 10
        out.loc[out["product_form"].astype(str).isin(["assay_solution", "reference_standard"]), "score"] -= 10
    if "page_type" in out.columns:
        out.loc[out["page_type"].astype(str).eq("product_page"), "score"] += 10
        out.loc[out["page_type"].astype(str).eq("search_page"), "score"] -= 20
    if "price_noise_flag" in out.columns:
        out.loc[_as_bool_series(out, "price_noise_flag"), "score"] -= 35
    if "stock_status" in out.columns:
        out.loc[out["stock_status"].astype(str).str.contains("visible|stock|available", case=False, na=False), "score"] += 6
    if "region" in out.columns:
        out.loc[out["region"].astype(str).str.contains("US", case=False, na=False), "score"] += 5
    if "product_url" in out.columns:
        out.loc[out["product_url"].notna(), "score"] += 5

    out["ranking_reason"] = out.apply(_reason, axis=1)
    sort_cols = [c for c in ["score", "bulk_estimate_eligible", "has_visible_price"] if c in out.columns]
    ascending = [False] * len(sort_cols)
    return out.sort_values(sort_cols, ascending=ascending) if sort_cols else out.sort_values("score", ascending=False)


def _reason(row: pd.Series) -> str:
    reasons = []
    if bool(row.get("cas_exact_match", False)):
        reasons.append("CAS confirmed")
    if bool(row.get("bulk_estimate_eligible", False)):
        reasons.append("curve-model eligible")
    elif bool(row.get("has_visible_price", False)):
        reasons.append("visible price")
    if str(row.get("price_pairing_confidence", "")) in {"HIGH", "MEDIUM"}:
        reasons.append(f"{row.get('price_pairing_confidence')} price pairing")
    if str(row.get("purity_pass", "")) == "Yes":
        reasons.append("purity passes")
    elif str(row.get("purity_pass", "")) == "No":
        reasons.append("below required purity")
    if str(row.get("product_form", "")) in {"assay_solution", "reference_standard"}:
        reasons.append(str(row.get("product_form")))
    if bool(row.get("price_noise_flag", False)):
        reasons.append("price noise rejected")
    if not reasons:
        reasons.append("supplier/source lead")
    return "; ".join(reasons)
