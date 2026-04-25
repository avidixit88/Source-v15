from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math
import numpy as np
import pandas as pd

UNIT_TO_GRAMS = {
    "ug": 0.000001,
    "µg": 0.000001,
    "mg": 0.001,
    "g": 1.0,
    "kg": 1000.0,
    "mL": None,
    "L": None,
}

Scenario = Literal["Conservative", "Base", "Aggressive"]
SCALING_EXPONENTS: dict[Scenario, float] = {"Conservative": 0.92, "Base": 0.82, "Aggressive": 0.72}

@dataclass(frozen=True)
class BulkEstimate:
    scenario: Scenario
    estimated_total_price: float
    estimated_unit_price_per_g: float
    discount_vs_anchor_pct: float
    confidence: str
    explanation: str

@dataclass(frozen=True)
class CatalogQuantityAnalysis:
    desired_qty_g: float
    recommended_model_qty_g: float
    reasonable_ceiling_g: float
    largest_catalog_pack_g: float
    observed_price_points: int
    observed_quantity_span: float
    max_safe_scale_multiple: float
    support_level: str
    rfq_required_for_desired: bool
    anchor_supplier: str
    anchor_title: str
    anchor_pack_g: float
    anchor_price_usd: float
    fitted_exponent: float
    curve_r2: float | None
    explanation: str


def quantity_to_grams(quantity: float, unit: str) -> float | None:
    multiplier = UNIT_TO_GRAMS.get(unit)
    if multiplier is None:
        return None
    return float(quantity) * multiplier


def grams_to_label(qty_g: float | None) -> str:
    if qty_g is None or not math.isfinite(float(qty_g)):
        return "Not available"
    qty_g = float(qty_g)
    if qty_g >= 1000:
        return f"{qty_g / 1000:g} kg"
    if qty_g >= 1:
        return f"{qty_g:g} g"
    if qty_g >= 0.001:
        return f"{qty_g * 1000:g} mg"
    return f"{qty_g * 1_000_000:g} µg"


def normalize_price_points(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["pack_size_g"] = out.apply(lambda r: quantity_to_grams(r.get("pack_size", 0), str(r.get("pack_unit", "g"))), axis=1)
    out["price_per_g"] = out.apply(
        lambda r: (float(r["listed_price_usd"]) / r["pack_size_g"])
        if pd.notna(r.get("listed_price_usd")) and r.get("pack_size_g") and r.get("pack_size_g") > 0 else None,
        axis=1,
    )
    out["has_visible_price"] = out["price_per_g"].notna()
    return out


def choose_anchor_price(price_points: pd.DataFrame, desired_qty_g: float) -> pd.Series | None:
    visible = price_points[price_points["has_visible_price"] & price_points["pack_size_g"].notna()].copy()
    if visible.empty:
        return None
    below = visible[visible["pack_size_g"] <= desired_qty_g]
    if not below.empty:
        return below.sort_values(["pack_size_g", "price_per_g"], ascending=[False, True]).iloc[0]
    return visible.sort_values(["pack_size_g", "price_per_g"], ascending=[False, True]).iloc[0]


def _fit_log_curve(points: pd.DataFrame) -> tuple[float, float, float | None]:
    clean = points[["pack_size_g", "listed_price_usd"]].dropna().drop_duplicates().copy()
    clean = clean[(clean["pack_size_g"] > 0) & (clean["listed_price_usd"] > 0)]
    if len(clean) < 2:
        row = clean.iloc[0]
        exponent = SCALING_EXPONENTS["Base"]
        intercept = math.log(float(row["listed_price_usd"])) - exponent * math.log(float(row["pack_size_g"]))
        return exponent, intercept, None
    x = np.log(clean["pack_size_g"].astype(float).to_numpy())
    y = np.log(clean["listed_price_usd"].astype(float).to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    slope = float(max(0.50, min(1.03, slope)))
    return slope, float(intercept), float(max(0, min(1, r2)))


def _unit_price_is_reasonable_curve(points: pd.DataFrame) -> bool:
    clean = points[["pack_size_g", "price_per_g"]].dropna().drop_duplicates().sort_values("pack_size_g")
    if len(clean) < 2:
        return True
    prices = clean["price_per_g"].astype(float).to_list()
    violations = sum(1 for prev, cur in zip(prices, prices[1:]) if cur > prev * 1.08)
    return violations <= max(0, len(prices) // 4)


def _safe_scale_multiple(points: pd.DataFrame, r2: float | None, monotonic: bool) -> tuple[float, str]:
    clean = points[["pack_size_g", "listed_price_usd"]].dropna().drop_duplicates()
    n = len(clean)
    if n == 0:
        return 1.0, "No catalog support"
    min_pack = float(clean["pack_size_g"].min())
    max_pack = float(clean["pack_size_g"].max())
    span = max(max_pack / min_pack, 1.0) if min_pack > 0 else 1.0
    if n >= 4 and monotonic and (r2 is None or r2 >= 0.90):
        mult = min(100.0, max(15.0, math.sqrt(span) * 5.0))
        support = "Strong catalog curve"
    elif n >= 3 and monotonic:
        mult = min(50.0, max(8.0, math.sqrt(span) * 3.0))
        support = "Moderate catalog curve"
    elif n >= 2:
        mult = min(15.0, max(4.0, math.sqrt(span) * 1.75))
        support = "Limited catalog curve"
    else:
        mult = 3.0
        support = "Single catalog point"
    if max_pack < 0.01:
        mult = min(mult, 8.0)
    elif max_pack < 0.10:
        mult = min(mult, 15.0)
    elif max_pack < 1.0:
        mult = min(mult, 50.0)
    return round(mult, 2), support


def analyze_catalog_quantity_support(price_points: pd.DataFrame, desired_qty_g: float) -> tuple[CatalogQuantityAnalysis | None, pd.DataFrame]:
    if price_points.empty or desired_qty_g <= 0:
        return None, pd.DataFrame()
    df = price_points.copy()
    if "catalog_estimate_eligible" in df.columns:
        df = df[df["catalog_estimate_eligible"].astype(bool)]
    else:
        df = df[df.get("has_visible_price", False).astype(bool)]
    df = df[df["pack_size_g"].notna() & df["listed_price_usd"].notna()]
    df = df[(df["pack_size_g"] > 0) & (df["listed_price_usd"] > 0)]
    if df.empty:
        return None, pd.DataFrame()
    curve_records: list[dict] = []
    group_cols = [c for c in ["supplier", "canonical_url", "page_title", "purity", "product_form"] if c in df.columns]
    for _, g in df.groupby(group_cols, dropna=False):
        g = g.drop_duplicates(subset=["pack_size_g", "listed_price_usd"]).copy()
        if g.empty:
            continue
        exponent, intercept, r2 = _fit_log_curve(g)
        monotonic = _unit_price_is_reasonable_curve(g)
        max_pack = float(g["pack_size_g"].max())
        min_pack = float(g["pack_size_g"].min())
        span = max_pack / min_pack if min_pack > 0 else 1.0
        mult, support = _safe_scale_multiple(g, r2, monotonic)
        ceiling = max_pack * mult
        target_qty = min(float(desired_qty_g), ceiling)
        estimated_total_at_target = math.exp(intercept) * math.pow(target_qty, exponent)
        estimated_total_at_desired = math.exp(intercept) * math.pow(float(desired_qty_g), exponent) if desired_qty_g <= ceiling else None
        anchor = g.sort_values(["pack_size_g", "price_per_g"], ascending=[False, True]).iloc[0]
        curve_records.append({
            "supplier": g["supplier"].iloc[0] if "supplier" in g.columns else "Unknown",
            "page_title": g["page_title"].iloc[0] if "page_title" in g.columns else "",
            "canonical_url": g["canonical_url"].iloc[0] if "canonical_url" in g.columns else "",
            "purity": g["purity"].iloc[0] if "purity" in g.columns else "",
            "product_form": g["product_form"].iloc[0] if "product_form" in g.columns else "solid/mass",
            "observed_price_points": int(len(g)),
            "min_catalog_pack_g": min_pack,
            "largest_catalog_pack_g": max_pack,
            "observed_quantity_span": round(span, 2),
            "fitted_exponent": round(exponent, 3),
            "curve_r2": round(r2, 3) if r2 is not None else None,
            "monotonic_unit_price_curve": bool(monotonic),
            "max_safe_scale_multiple": mult,
            "reasonable_ceiling_g": round(ceiling, 6),
            "recommended_model_qty_g": round(target_qty, 6),
            "rfq_required_for_desired": bool(desired_qty_g > ceiling),
            "estimated_total_at_recommended_qty": round(estimated_total_at_target, 2),
            "estimated_total_at_desired_qty": round(estimated_total_at_desired, 2) if estimated_total_at_desired is not None else None,
            "support_level": support,
            "anchor_pack_g": float(anchor["pack_size_g"]),
            "anchor_price_usd": float(anchor["listed_price_usd"]),
            "anchor_supplier": str(anchor.get("supplier", "Unknown")),
        })
    curve_df = pd.DataFrame(curve_records)
    if curve_df.empty:
        return None, curve_df
    curve_df["_support_rank"] = curve_df["support_level"].map({"Strong catalog curve": 4, "Moderate catalog curve": 3, "Limited catalog curve": 2, "Single catalog point": 1}).fillna(0)
    curve_df = curve_df.sort_values(["rfq_required_for_desired", "reasonable_ceiling_g", "_support_rank", "observed_price_points", "largest_catalog_pack_g"], ascending=[True, False, False, False, False]).drop(columns=["_support_rank"])
    best = curve_df.iloc[0]
    rfq_required = bool(best["rfq_required_for_desired"])
    explanation = (
        f"The strongest public catalog curve is from {best['supplier']} with {int(best['observed_price_points'])} bound quantity/price points. "
        f"Largest public pack is {grams_to_label(float(best['largest_catalog_pack_g']))}. "
        f"Based on the observed pack-size span and curve quality, v15 treats {grams_to_label(float(best['reasonable_ceiling_g']))} as the largest rationally modelable catalog quantity."
    )
    if rfq_required:
        explanation += f" The requested {grams_to_label(desired_qty_g)} is beyond that ceiling, so the requested amount is RFQ territory."
    else:
        explanation += f" The requested {grams_to_label(desired_qty_g)} is inside that ceiling, so a catalog-curve estimate is reasonable but still not a confirmed quote."
    analysis = CatalogQuantityAnalysis(
        desired_qty_g=float(desired_qty_g),
        recommended_model_qty_g=float(best["recommended_model_qty_g"]),
        reasonable_ceiling_g=float(best["reasonable_ceiling_g"]),
        largest_catalog_pack_g=float(best["largest_catalog_pack_g"]),
        observed_price_points=int(best["observed_price_points"]),
        observed_quantity_span=float(best["observed_quantity_span"]),
        max_safe_scale_multiple=float(best["max_safe_scale_multiple"]),
        support_level=str(best["support_level"]),
        rfq_required_for_desired=rfq_required,
        anchor_supplier=str(best["anchor_supplier"]),
        anchor_title=str(best["page_title"]),
        anchor_pack_g=float(best["anchor_pack_g"]),
        anchor_price_usd=float(best["anchor_price_usd"]),
        fitted_exponent=float(best["fitted_exponent"]),
        curve_r2=float(best["curve_r2"]) if pd.notna(best["curve_r2"]) else None,
        explanation=explanation,
    )
    return analysis, curve_df


def build_catalog_estimate_scenarios(analysis: CatalogQuantityAnalysis, curve_df: pd.DataFrame) -> pd.DataFrame:
    if analysis is None or curve_df.empty:
        return pd.DataFrame()
    best = curve_df.iloc[0]
    base_exp = float(best["fitted_exponent"])
    target_qty = float(analysis.recommended_model_qty_g)
    if target_qty <= 0:
        return pd.DataFrame()
    exponent_map = {"Conservative": min(1.0, base_exp + 0.08), "Base": base_exp, "Aggressive": max(0.50, base_exp - 0.08)}
    anchor_qty = float(best["anchor_pack_g"])
    anchor_price = float(best["anchor_price_usd"])
    rows = []
    for scenario, exponent in exponent_map.items():
        estimated_total = anchor_price * math.pow(target_qty / anchor_qty, exponent)
        anchor_unit = anchor_price / anchor_qty
        unit = estimated_total / target_qty
        rows.append({
            "scenario": scenario,
            "target_quantity": grams_to_label(target_qty),
            "target_quantity_g": round(target_qty, 6),
            "estimated_total_price": round(estimated_total, 2),
            "estimated_unit_price_per_g": round(unit, 4),
            "discount_vs_anchor_pct": round((1 - unit / anchor_unit) * 100, 1),
            "confidence": analysis.support_level,
            "rfq_required_for_original_desired_qty": analysis.rfq_required_for_desired,
            "explanation": "Empirical catalog-curve estimate using bound quantity/price rows. Not a confirmed quote.",
        })
    return pd.DataFrame(rows)


def estimate_bulk_price(anchor_pack_g: float, anchor_total_price: float, desired_qty_g: float, scenario: Scenario, visible_price_points: int) -> BulkEstimate:
    if anchor_pack_g <= 0 or anchor_total_price <= 0 or desired_qty_g <= 0:
        raise ValueError("anchor_pack_g, anchor_total_price, and desired_qty_g must be positive")
    exponent = SCALING_EXPONENTS[scenario]
    ratio = desired_qty_g / anchor_pack_g
    estimated_total = anchor_total_price * math.pow(ratio, exponent)
    estimated_unit = estimated_total / desired_qty_g
    anchor_unit = anchor_total_price / anchor_pack_g
    discount_pct = (1 - (estimated_unit / anchor_unit)) * 100
    if visible_price_points >= 3:
        confidence = "Medium"
        explanation = "Multiple visible pack prices exist, so the curve has some support. Confirm with RFQ before purchasing."
    elif visible_price_points == 2:
        confidence = "Low-Medium"
        explanation = "Only two visible price points exist. Treat as directional until supplier confirms bulk pricing."
    else:
        confidence = "Low"
        explanation = "Only one visible price point exists. This is a rough catalog-to-bulk estimate, not a confirmed quote."
    return BulkEstimate(scenario, round(estimated_total, 2), round(estimated_unit, 4), round(discount_pct, 1), confidence, explanation)
